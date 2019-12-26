#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import glob
import random

import imgaug
from imgaug import augmenters as iaa
from PIL import Image
import matplotlib.pyplot as plt


import numpy as np 
import tensorflow as tf
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D,AveragePooling2D, ZeroPadding2D, concatenate,Concatenate, UpSampling2D, Activation, Lambda
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # noqa

import sklearn.metrics
import io
import itertools
import tqdm
from six.moves import range

import time
import cv2
import tifffile 

from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

from .models.densenet import *
from .models.inception import *
from .models.deeplabv3 import *

from .helpers.utils import *
from .loaders.dataloader import *

from os.path import expanduser
home = expanduser("~")


# Random Seeds
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# get_prediction
def get_prediction(wsi_path, 
                                   mask_path=None, 
                                   label_path=None, 
                                   batch_size=64, 
                                   models=None, 
                                   tta_list=None,
                                   num_workers=8, 
                                   verbose=0, 
                                   patch_size = 256,
                                   stride_size = 256,
                                   mask_level = -1,
                                   status = None):
        """
                        patch based segmentor
        """

        dataset_obj = WSIStridedPatchDataset(wsi_path, 
                                                                                mask_path,
                                                                                label_path,
                                                                                image_size=patch_size,
                                                                                normalize=True,
                                                                                flip=None, rotate=None,
                                                                                sampling_stride=stride_size, 
                                                                                mask_level=mask_level,
                                                                                roi_masking=True)


        dataloader = DataLoader(dataset_obj, batch_size=batch_size, num_workers=num_workers, drop_last=True)
        print ("Length of DataLoader: {}".format(len(dataloader)))

        if tta_list == None:
                tta_list = np.array(['DEFAULT'])
        else:
                tta_list = np.array(tta_list)
                tta_list = np.concatenate([np.array(['DEFAULT']), tta_list])
                
        probs_map = {}
        
        eps = 0.0001
        num_batch  = len(dataloader)
        batch_size = dataloader.batch_size
        map_x_size = dataloader.dataset._slide.level_dimensions[0][0]
        map_y_size = dataloader.dataset._slide.level_dimensions[0][1]
        factor = dataloader.dataset._sampling_stride
        flip   = dataloader.dataset._flip
        rotate = dataloader.dataset._rotate 

        digipathai_folder = os.path.join(home, '.DigiPathAI')
        memmaps_path = os.path.join(digipathai_folder,'memmaps')
        os.makedirs(memmaps_path,exist_ok=True)

        probs_map['mean'] = np.memmap(os.path.join(memmaps_path,'%s.dat'%('mean')), 
                                                                dtype=np.float32,
                                                                mode='w+', 
                                                                shape=(dataloader.dataset._slide.level_dimensions[0]))

        probs_map['var'] = np.memmap(os.path.join(memmaps_path,'%s.dat'%('var')), 
                                                                dtype=np.float32,
                                                                mode='w+', 
                                                                shape=(dataloader.dataset._slide.level_dimensions[0]))

        count_map = np.memmap(os.path.join(memmaps_path,'%s.dat'%('count_map')), 
                                                                dtype=np.uint8,
                                                                mode='w+', 
                                                                shape=(dataloader.dataset._slide.level_dimensions[0]))

        count_map[:, :] = 0
        # for i, model_name in enumerate(models.keys()):
        #       probs_map[model_name] = np.zeros(dataloader.dataset._slide.level_dimensions[0])

        with tqdm(total=len(dataloader)) as pbar:
            for ii, (image_patches, x_coords, y_coords, label_patches) in enumerate(dataloader):
                    pbar.update(1)
                    if status is not None:
                            status['progress'] = int(ii*100.0/ (len(models.keys())*len(dataloader)))


                    image_patches = image_patches.cpu().data.numpy()
                    label_patches = label_patches.cpu().data.numpy()
                    x_coords = x_coords.cpu().data.numpy()
                    y_coords = y_coords.cpu().data.numpy()
                    
                    
                    
                    patch_predictions = []
                    for tta_ in tta_list:
                            image_patches = apply_tta(image_patches, tta_)

                            for j, model_name in enumerate(models.keys()):
                                    prediction = models[model_name].predict(image_patches, 
                                                                                                               batch_size=batch_size, 
                                                                                                               verbose=verbose, steps=None)
                                    try: 
                                            prediction_trans = transform_prob(prediction, tta_)
                                            patch_predictions.append(prediction_trans)
                                    except: continue

                    patch_predictions = np.array(patch_predictions)

                    for i in range(batch_size):
                            shape = patch_predictions[0, 0].shape
                            probs_map['mean'][x_coords[i]: x_coords[i]+shape[0] , 
                                            y_coords[i]: y_coords[i]+shape[1]]  += np.mean(patch_predictions, axis=0)[i,:,:,1]

                            probs_map['var'][x_coords[i]: x_coords[i]+shape[0] , 
                                            y_coords[i]: y_coords[i]+shape[1]]  += np.var(patch_predictions, axis=0)[i, :,:,1]

                            count_map[x_coords[i]: x_coords[i]+shape[0], 
                                             y_coords[i]: y_coords[i]+shape[1]] += np.ones_like(patch_predictions[0, 0, : ,:, 1],dtype=np.uint8)
        
        np.place(count_map, count_map == 0, 1)
        probs_map['mean'] /= count_map
        probs_map['var']  /= count_map**2.0
        #TTD
        # if label_path:
                # return (dataset_obj.get_image(),
                                        # probs_map,
                                        # dataset_obj.get_mask()/255, 
                                        # dataset_obj.get_label_mask())
        # else:
                # return (dataset_obj.get_image(),
                                        # probs_map,
                                        # dataset_obj.get_mask()/255, 
                                        # np.zeros(count_map.shape).astype('uint8'))
        return (dataset_obj.get_image(),probs_map)

   
def getSegmentation(img_path, 
                        patch_size  = 256, 
                        stride_size = 128,
                        batch_size  = 32,
                        tta_list    = None,
                        crf         = False,
                        mask_path   = '../Results',
                        uncertainty_path   = '../Results',
                        status      = None,
                        quick       = True,
                        mask_level       = -1,
                        model       = 'dense',
                        mode        = 'colon'):
        """
                        args:
                                img_path: WSI tiff image path (str)
                                patch_size: patch size for inference (int)
                                stride_size: stride to skip during segmentation (int)
                                batch_size: batch_size during inference (int)
                                quick: if True; final segmentation is ensemble of 4 different models
                                                else: prediction is of single model (bool)
                                tta_list: type of augmentation required/examples/colon-cancer-1-slide.tiff# during inference
                                                 allowed: ['FLIP_LEFT_RIGHT', 'ROTATE_90', 'ROTATE_180', 'ROTATE_270'] (list(str))
                                crf: application of conditional random fields in post processing step (bool)
                                save_path: path to save final segmentation mask (str)
                                status: required for webserver (json)
                                mode: tissue type

                        return :
                                saves the prediction in given path (in .tiff format)
                                prediction: predicted segmentation mask

        """

        mode = mode.lower()
        print ("==================================================")
        print (mode)
        if mode not in ['colon', 'liver', 'breast']: raise ValueError("Unknown mode found, allowed fields are: ['colon', 'liver', 'breast']")

        if mode == 'colon':
                path = os.path.join(home, '.DigiPathAI/digestpath_models')
                if (not os.path.exists(os.path.join(path, 'digestpath_inception.h5'))) or \
                        (not os.path.exists(os.path.join(path, 'digestpath_deeplabv3.h5'))) or \
                        (not os.path.exists(os.path.join(path, 'digestpath_densenet.h5'))):
                        if status is not None: status['status'] = "Downloading Trained Models"
                        download_digestpath() 
                        model_path_inception = os.path.join(path, 'digestpath_inception.h5')
                        model_path_deeplabv3 = os.path.join(path, 'digestpath_deeplabv3.h5')
                        model_path_densenet = os.path.join(path, 'digestpath_densenet.h5')
                else :
                        if status is not None: status['status'] = "Found Trained Models, Skipping download"
                        model_path_inception = os.path.join(path, 'digestpath_inception.h5')
                        model_path_deeplabv3 = os.path.join(path, 'digestpath_deeplabv3.h5')
                        model_path_densenet = os.path.join(path, 'digestpath_densenet.h5')

        elif mode == 'liver':
                path = os.path.join(home, '.DigiPathAI/paip_models')
                if (not os.path.exists(os.path.join(path, 'paip_inception.h5'))) or \
                        (not os.path.exists(os.path.join(path, 'paip_deeplabv3.h5'))) or \
                        (not os.path.exists(os.path.join(path, 'paip_densenet.h5'))):
                        if status is not None: status['status'] = "Downloading Trained Models"
                        download_paip() 
                        model_path_inception = os.path.join(path, 'paip_inception.h5')
                        model_path_deeplabv3 = os.path.join(path, 'paip_deeplabv3.h5')
                        model_path_densenet  = os.path.join(path, 'paip_densenet.h5')
                else :
                        if status is not None: status['status'] = "Found Trained Models, Skipping download"
                        model_path_inception = os.path.join(path, 'paip_inception.h5')
                        model_path_deeplabv3 = os.path.join(path, 'paip_deeplabv3.h5')
                        model_path_densenet  = os.path.join(path, 'paip_densenet.h5')

        elif mode == 'breast':
                path = os.path.join(home, '.DigiPathAI/camelyon_models')
                if (not os.path.exists(os.path.join(path, 'camelyon_inception.h5'))) or \
                        (not os.path.exists(os.path.join(path, 'camelyon_deeplabv3.h5'))) or \
                        (not os.path.exists(os.path.join(path, 'camelyon_densenet.h5'))):
                        if status is not None: status['status'] = "Downloading Trained Models"
                        download_camelyon() 
                        model_path_inception = os.path.join(path, 'camelyon_inception.h5')
                        model_path_deeplabv3 = os.path.join(path, 'camelyon_deeplabv3.h5')
                        model_path_densenet = os.path.join(path, 'camelyon_densenet.h5')
                else :
                        if status is not None: status['status'] = "Found Trained Models, Skipping download"
                        model_path_inception = os.path.join(path, 'camelyon_inception.h5')
                        model_path_deeplabv3 = os.path.join(path, 'camelyon_deeplabv3.h5')
                        model_path_densenet  = os.path.join(path, 'camelyon_densenet.h5')


        if status is not None: status['status'] = "Loading Trained weights"
        core_config = tf.ConfigProto()
        core_config.gpu_options.allow_growth = True 
        session =tf.Session(config=core_config) 
        K.set_session(session)

        print ("---------------------- {}, {} ---------------".format(model, quick))
        if not quick:
                models_to_consider = {'dense': model_path_densenet, 
                                                  'inception': model_path_inception, 
                                                  'deeplabv3': model_path_deeplabv3}
        else:
                if model == 'dense':
                        models_to_consider = {'dense': model_path_densenet}
                elif model == 'inception':
                        models_to_consider = {'inception': model_path_inception}
                elif model == 'deeplabv3':
                        models_to_consider = {'deeplabv3': model_path_deeplabv3}
                else:
                        raise ValueError("Unknown model provided, allowed models ['dense', 'inception', 'deeplabv3']")



        models = {}
        for i, model_name in enumerate(models_to_consider.keys()):
                models[model_name] = load_trained_models(model_name, 
                                                                models_to_consider[model_name],
                                                                patch_size = patch_size)

        threshold = 0.3

        if status is not None: status['status'] = "Running segmentation"
        img, probs_map = get_prediction(img_path, 
                                                                        mask_path = None, 
                                                                        mask_level = mask_level,
                                                                        label_path = None,
                                                                        batch_size = batch_size,
                                                                        tta_list = tta_list,
                                                                        models = models,
                                                                        patch_size = patch_size,
                                                                        stride_size = stride_size,
                                                                        status = status)
        
        # for key in probs_map.keys():
        #       probs_map[key] = BinMorphoProcessMask(probs_map[key])
        
        # if crf:
                # pred = post_process_crf(img, np.concatenate([1 - probs_map['mean'].T[...,None], 
                                                                                                        # probs_map['mean'].T[..., None]], 
                                                                                                        # axis = -1) , 2)
        # else :
        np.place(probs_map['mean'],probs_map['mean'] >= threshold,255)
        np.place(probs_map['mean'],probs_map['mean'] < threshold,0)
        
        if status is not None:
                status['progress'] = 100
        
        if status is not None:
                status['status'] = "Saving Prediction Mask..."

        tifffile.imsave(mask_path, probs_map['mean'].T, compress=9)
        os.system('convert ' + mask_path + " -compress jpeg -quality 90 -define tiff:tile-geometry=256x256 ptif:"+mask_path)
        # os.system("free -h")

        if status is not None:
                status['status'] = "Saving Prediction Uncertanity..."
        tifffile.imsave(uncertainty_path, probs_map['var'].T*255, compress=9)
        os.system('convert ' + uncertainty_path + " -compress jpeg -quality 90 -define tiff:tile-geometry=256x256 ptif:"+uncertainty_path)
        
        if status is not None:
                status['progress'] = 0
        return np.array(probs_map['mean'])
