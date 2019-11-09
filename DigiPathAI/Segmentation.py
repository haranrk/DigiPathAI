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
from tqdm import tqdm
import matplotlib.pyplot as plt


import numpy as np 
import tensorflow as tf
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, BatchNormalization, Conv2D, MaxPooling2D,                             						
					AveragePooling2D, ZeroPadding2D, concatenate, 	
					Concatenate, UpSampling2D, Activation, Lambda)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # noqa

import sklearn.metrics
import io
import itertools
from six.moves import range

import time
import cv2
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

from models.densenet import *
from models.inception import *
from models.deeplabv3 import *

from helpers.utils import *
from loaders.dataloader import *



# Random Seeds
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)


os.environ["CUDA_VISIBLE_DEVICES"] = '0'




# get_prediction
def get_prediction(wsi_path, 
				   mask_path=None, 
				   label_path=None, 
				   batch_size=8, 
				   models=None, 
				   tta_list=None,
				   num_workers=8, 
				   verbose=0, 
				   patch_size = 256,
				   stride_size = 256,
				   status = None):
	"""
	"""
	dataset_obj = WSIStridedPatchDataset(wsi_path, 
										mask_path,
										label_path,
										image_size=patch_size,
										normalize=True,
										flip=None, rotate=None,
										sampling_stride=stride_size, roi_masking=True)


	dataloader = DataLoader(dataset_obj, batch_size=batch_size, num_workers=num_workers, drop_last=True)
	print (dataloader.dataset.__len__(), dataloader.__len__())
	
	if tta_list == None:
		tta_list = np.array(['DEFAULT'])
	else:
		tta_list = np.array(tta_list)
		tta_list = np.concatenate([np.array(['DEFAULT']), tta_list])
		
	probs_map = {}
	count_map = np.zeros(dataloader.dataset._mask.shape)
	
	eps = 0.0001
	num_batch  = len(dataloader)
	batch_size = dataloader.batch_size
	map_x_size = dataloader.dataset._mask.shape[0]
	map_y_size = dataloader.dataset._mask.shape[1]
	factor = dataloader.dataset._sampling_stride
	flip   = dataloader.dataset._flip
	rotate = dataloader.dataset._rotate 
		
	
	for i, model_name in enumerate(models.keys()):
		probs_map[model_name] = np.zeros(dataloader.dataset._mask.shape)

	for i, (image_patches, x_coords, y_coords, label_patches) in enumerate(dataloader):
		
		if status is not None:
			status['progress'] = int(i*100.0/ len(dataloader))
			print ("========================", status['progress'])

		image_patches = image_patches.cpu().data.numpy()
		label_patches = label_patches.cpu().data.numpy()
		x_coords = x_coords.cpu().data.numpy()
		y_coords = y_coords.cpu().data.numpy()
		
		for j, model_name in enumerate(models.keys()):
			
			for tta_ in tta_list:
				image_patches = apply_tta(image_patches, tta_)
			
				prediction = models[model_name].predict(image_patches, 
													   batch_size=batch_size, 
													   verbose=verbose, steps=None)
				for i in range(batch_size):
					prediction_trans = transform_prob(prediction[i], tta_)/(1.*len(tta_list))
					shape = prediction_trans.shape
					probs_map[model_name][x_coords[i]-shape[0]//2: x_coords[i]+shape[0]//2 , 
						  y_coords[i]-shape[1]//2: y_coords[i]+shape[1]//2]  += prediction_trans[:,:,1]
 
					if j == 0:
						count_map[x_coords[i]-shape[0]//2: x_coords[i]+shape[0]//2, 
								 y_coords[i]-shape[1]//2: y_coords[i]+shape[1]//2] += np.ones_like(prediction[0,: ,:,1])

	
	if label_path:
		return (dataset_obj.get_image(),
					probs_map,
					count_map, 
					dataset_obj.get_mask()/255, 
					dataset_obj.get_label_mask()/255)
	else:
		return (dataset_obj.get_image(),
					probs_map,
					count_map, 
					dataset_obj.get_mask()/255, 
					np.zeros(count_map.shape).astype('uint8'))



   
def predictImage(img_path, 
			patch_size  = 256, 
			stride_size = 128,
			batch_size  = 32,
			quick       = True,
			tta_list    = None,
			crf         = False,
			save_path   = '../Results',
			status      = None):
	"""
	 ['FLIP_LEFT_RIGHT', 'ROTATE_90', 'ROTATE_180', 'ROTATE_270']
	"""
	model_path_inception = '/home/pi/Projects/DigiPathAI/model_weights/inception.h5'
	model_path_deeplabv3 = '/home/pi/Projects/DigiPathAI/model_weights/deeplabv3.h5'
	model_path_densenet2 = '/home/pi/Projects/DigiPathAI/model_weights/densenet_fold2.h5'
	model_path_densenet1 = '/home/pi/Projects/DigiPathAI/model_weights/densenet_fold1.h5'



	core_config = tf.ConfigProto()
	core_config.gpu_options.allow_growth = True 
	session =tf.Session(config=core_config) 
	K.set_session(session)

	if not quick:
		models_to_consider = {'dense1': model_path_densenet1,
						  'dense2': model_path_densenet2, 
						  'inception': model_path_inception, 
						  'deeplabv3': model_path_deeplabv3}
	else:
		models_to_consider = {'dense1': model_path_densenet1}

	models = {}
	for i, model_name in enumerate(models_to_consider.keys()):
		models[model_name] = load_trained_models(model_name, 
								models_to_consider[model_name])

	threshold = 0.5

	if not os.path.exists(os.path.join(save_path)):
		os.makedirs(os.path.join(save_path))
	
	img, probs_map, count_map, tissue_mask, label_mask  = get_prediction(img_path, 
									mask_path = None, 
									label_path = None,
					  			        batch_size = batch_size,
									tta_list = tta_list,
									models = models,
									patch_size = patch_size,
									stride_size = stride_size,
									status = status)
	count_map[count_map == 0] = 1
	for key in probs_map.keys():
		probs_map[key] = BinMorphoProcessMask(probs_map[key]*(tissue_mask>0).astype('float'))
			
	mean_probs, uncertanity = get_mean_img(probs_map.values(), count_map)
	mean_probs = mean_probs[..., None]
	
	if crf:
		pred = post_process_crf(img, np.concatenate([1-mean_probs, mean_probs], axis = -1) , 2)
	else :
		pred = mean_probs[:, :, 0] > threshold

	
	img_name = img_path.split("/")[-1]
	pred = Image.fromarray(pred.astype('uint8'))
	pred.save(save_path.replace("tiff", 'jpg'))
	os.system(save_path.replace("tiff", 'jpg')+ " -compress jpeg -quality 90 -define tiff:tile-geometry=2738x2847 ptif:"+save_path)
	return np.array(pred)


if __name__ == "__main__":

	# create session with models
	model_path_inception = '../model_weights/inception.h5'
	model_path_deeplabv3 = '../model_weights/deeplabv3.h5'
	model_path_densenet2 = '../model_weights/densenet_fold2.h5'
	model_path_densenet1 = '../model_weights/densenet_fold1.h5'

	patch_size = 256
	stride_size = 128
	batch_size = 64
	core_config = tf.ConfigProto()
	core_config.gpu_options.allow_growth = True 
	session =tf.Session(config=core_config) 
	K.set_session(session)


	from glob import glob
	models_to_consider = {'dense1': model_path_densenet1,
						  'dense2': model_path_densenet2, 
						  'inception': model_path_inception, 
						  'deeplabv3': model_path_deeplabv3}
	models = {}
	for i, model_name in enumerate(models_to_consider.keys()):
		models[model_name] = load_trained_models(model_name, 
												 models_to_consider[model_name])


	# all_imgs = glob('/media/'+whoami+'/Kori/histopath/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos/*.jpg')
	all_imgs = glob('input/*.jpg')
	all_imgs = [pth for pth in all_imgs if not pth.__contains__('mask')][2:]

	results_root = 'Task2/Results/'

	if not os.path.exists(os.path.join(results_root, 'predictions')):
		os.makedirs(os.path.join(results_root, 'predictions'))


	threshold = 0.5
	tta_list = None #['ROTATE_90', 'ROTATE_180', 'ROTATE_270'] # ['FLIP_LEFT_RIGHT', 'ROTATE_90', 'ROTATE_180', 'ROTATE_270']

	img_path = []; label = [];

	for img in all_imgs:
		wsi_path = img
		label_path = None #img.split('.')[0] + '_mask.jpg'
		mask_path  = None
		time_now = time.time()
		img, probs_map, count_map, tissue_mask, gt = get_prediction(wsi_path, mask_path, 
														   label_path,
														   batch_size = batch_size,
														   tta_list = tta_list,
														   models = models)

		
		st = time
		count_map[count_map == 0] = 1
		for key in probs_map.keys():
			probs_map[key] = BinMorphoProcessMask(probs_map[key]*(tissue_mask>0).astype('float'))
			
		mean_probs = get_mean_img(probs_map.values(), count_map)[..., None]

		crf = post_process_crf(img, np.concatenate([1-mean_probs, mean_probs], axis = -1) , 2)
		mean_probs = mean_probs[:, :, 0]
		tmap = (count_map > 1).astype('float')
		tbr = np.mean(np.sum(crf)/np.sum(tmap))
		
		time_spent = time.time() - time_now
		print("Time Spent {}".format(time_spent))
		


		for key in probs_map.keys():
			print (key + ": " + str(iou( probs_map[key]/count_map > threshold, gt.astype('bool'))))

		print ("Mean : " + str(iou( mean_probs > threshold, gt.astype('bool'))))
		print ("CRF :  " + str(iou( crf > threshold, gt.astype('bool'))))
		print ("tbr : " + str(tbr))

	
		if (np.mean(uncertainity) > 0.5*1e-3) and (tbr < 0.005): label.append(0.0)
		else: label.append(1.0)

		img_name = img.split("/")[-1]
		img_path.append(img_name)

		crf = Image.fromarray(crf.astype('uint8'))
		crf.save(os.path.join(results_root, 'predictions', img_name))

		# imshow(count_map, gt, *[t>threshold for t in probs_map.values()], mean_probs>threshold,  crf > threshold)


	df = pd.DataFrame()
	df['image_name'] = img_path
	df['score'] = label
	df.to_csv(os.path.join(results_root, 'predict.csv'), index=False)
