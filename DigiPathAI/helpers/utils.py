from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import glob
import random

from ..models.densenet import *
from ..models.inception import *
from ..models.deeplabv3 import *

import imgaug
from imgaug import augmenters as iaa
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


import numpy as np 
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, BatchNormalization, Conv2D, MaxPooling2D,                             						AveragePooling2D, ZeroPadding2D, concatenate, 	
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

import openslide
import time
import cv2
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import wget

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian
from os.path import expanduser
home = expanduser("~")
	

def download_digestpath():
    """
        Downloads nobrainer models for transferleraning
    """
    model_path = os.path.join(home, '.DigiPathAI/digestpath_models')
    if (not os.path.exists(model_path)) or (len(os.listdir(model_path)) == 0):
        os.makedirs(model_path, exist_ok=True)
        wget.download('https://github.com/haranrk/DigiPathAI/releases/download/models/digestpath_deeplabv3.h5',
                        out = model_path)
        wget.download('https://github.com/haranrk/DigiPathAI/releases/download/models/digestpath_densenet_fold1.h5',
                        out = model_path)
        wget.download('https://github.com/haranrk/DigiPathAI/releases/download/models/digestpath_densenet_fold2.h5',
                        out = model_path)
        wget.download('https://github.com/haranrk/DigiPathAI/releases/download/models/digestpath_inception.h5',
                        out = model_path)


# Image Visualize Helper Functions
def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    axis_off = kwargs.get('axis_off','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
            if axis_off: 
              plt.axis('off')  
    plt.show()

    

# Image Helper Functions
def imsave(*args, **kwargs):
     """
     Concatenate the images given in args and saves them as a single image in the specified output destination.
     Images should be numpy arrays and have same dimensions along the 0 axis.
     imsave(im1,im2,out="sample.png")
     """
     args_list = list(args)
     for i in range(len(args_list)):
         if type(args_list[i]) != np.ndarray:
             print("Not a numpy array")
             return 0
         if len(args_list[i].shape) == 2:
             args_list[i] = np.dstack([args_list[i]]*3)
             if args_list[i].max() == 1:
                args_list[i] = args_list[i]*255

     out_destination = kwargs.get("out",'')
     try:
         concatenated_arr = np.concatenate(args_list,axis=1)
         im = Image.fromarray(np.uint8(concatenated_arr))
     except Exception as e:
         print(e)
         import ipdb; ipdb.set_trace()
         return 0
     if out_destination:
         print("Saving to %s"%(out_destination))
         im.save(out_destination)
     else:
        return im


def normalize_minmax(data):
    """
    Normalize contrast across volume
    """
    _min = np.float(np.min(data))
    _max = np.float(np.max(data))
    if (_max-_min)!=0:
        img = (data - _min) / (_max-_min)
    else:
        img = np.zeros_like(data)            
    return img


# In[5]:







# pre processing helpers

def BinMorphoProcessMask(mask):
    """
    Binary operation performed on tissue mask
    """
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    return image_open 


def BinMorphoProcessMaskOS(mask, level):
    """
    Binary operation performed on tissue mask
    """
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    if level == 2:
        kernel = np.ones((60, 60), dtype=np.uint8)
    elif level == 3:
        kernel = np.ones((35, 35), dtype=np.uint8)
    else:
        raise ValueError
    image = cv2.dilate(image_open,kernel,iterations = 1)
    return image

def get_bbox(cont_img, rgb_image=None):
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if rgb_image is not None:
        rgb_contour = rgb_image.copy()
        line_color = (0, 0, 255)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    for x, y, h, w in bounding_boxes:
        rgb_contour = cv2.rectangle(rgb_contour,(x,y),(x+h,y+w),(0,255,0),2)
    return bounding_boxes, rgb_contour

def get_all_bbox_masks(mask, stride_factor):
    """
    Find the bbox and corresponding masks
    """
    bbox_mask = np.zeros_like(mask)
    bounding_boxes, _ = get_bbox(mask)
    y_size, x_size = bbox_mask.shape
    for x, y, h, w in bounding_boxes:
        x_min = x - stride_factor
        x_max = x + h + stride_factor
        y_min = y - stride_factor
        y_max = y + w + stride_factor
        if x_min < 0: 
         x_min = 0
        if y_min < 0: 
         y_min = 0
        if x_max > x_size: 
         x_max = x_size - 1
        if y_max > y_size: 
         y_max = y_size - 1      
        bbox_mask[y_min:y_max, x_min:x_max]=1
    return bbox_mask

def get_all_bbox_masks_with_stride(mask, stride_factor):
    """
    Find the bbox and corresponding masks
    """
    bbox_mask = np.zeros_like(mask)
    bounding_boxes, _ = get_bbox(mask)
    y_size, x_size = bbox_mask.shape
    for x, y, h, w in bounding_boxes:
        x_min = x - stride_factor
        x_max = x + h + stride_factor
        y_min = y - stride_factor
        y_max = y + w + stride_factor
        if x_min < 0: 
            x_min = 0
        if y_min < 0: 
            y_min = 0
        if x_max > x_size: 
            x_max = x_size - 1
        if y_max > y_size: 
            y_max = y_size - 1      
        bbox_mask[y_min:y_max:stride_factor, x_min:x_max:stride_factor]=1
        
    return bbox_mask

def find_largest_bbox(mask, stride_factor):
    """
    Find the largest bounding box encompassing all the blobs
    """
    y_size, x_size = mask.shape
    x, y = np.where(mask==1)
    bbox_mask = np.zeros_like(mask)
    x_min = np.min(x) - stride_factor
    x_max = np.max(x) + stride_factor
    y_min = np.min(y) - stride_factor
    y_max = np.max(y) + stride_factor
    
    if x_min < 0: 
        x_min = 0
    
    if y_min < 0: 
        y_min = 0

    if x_max > x_size: 
        x_max = x_size - 1
    
    if y_min > y_size: 
        y_max = y_size - 1    
    
    bbox_mask[x_min:x_max, y_min:y_max]=1
    return bbox_mask
    
# Functions
def ReadWholeSlideImage(image_path):
    img = Image.open(image_path)
    return img

def getImagePatch(image, coords, size):
    image = np.array(image)
    (x, y) = coords
    if len(image.shape) == 3:
        return image[x-size//2:x+size//2, y-size//2:y+size//2, :]
    else:
        return image[x-size//2:x+size//2, y-size//2:y+size//2]
    
def TissueMaskGeneration(slide_obj, RGB_min=50):
    img_RGB = np.array(slide_obj)
    img_HSV = rgb2hsv(img_RGB)
    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return tissue_mask


def TissueMaskGenerationOS(slide_obj, level, RGB_min=50):
    img_RGB = slide_obj.read_region((0, 0),level,slide_obj.level_dimensions[level])
    img_RGB = np.transpose(np.array(img_RGB.convert('RGB')),axes=[1,0,2])
    img_HSV = rgb2hsv(img_RGB)
    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    # r = img_RGB[:,:,0] < 235
    # g = img_RGB[:,:,1] < 210
    # b = img_RGB[:,:,2] < 235
    # tissue_mask = np.logical_or(r,np.logical_or(g,b))
    return tissue_mask 

    
def TissueMaskGeneration_BINOS(slide_obj, level):
    img_RGB = np.transpose(np.array(slide_obj.read_region((0, 0),
                       level,
                       slide_obj.level_dimensions[level]).convert('RGB')),
                       axes=[1, 0, 2])    
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
    img_S = img_HSV[:, :, 1]
    _,tissue_mask = cv2.threshold(img_S, 0, 255, cv2.THRESH_BINARY)
    return np.array(tissue_mask)

def TissueMaskGeneration_BIN_OTSUOS(slide_obj, level):
    img_RGB = np.transpose(np.array(slide_obj.read_region((0, 0),
                       level,
                       slide_obj.level_dimensions[level]).convert('RGB')),
                       axes=[1, 0, 2])    
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
    img_S = img_HSV[:, :, 1]
    _,tissue_mask = cv2.threshold(img_S, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return np.array(tissue_mask)

def TissueMaskGenerationPatch(patchRGB):
    '''
    Returns mask of tissue that obeys the threshold set by paip
    '''
    r = patchRGB[:,:,0] < 235
    g = patchRGB[:,:,1] < 210
    b = patchRGB[:,:,2] < 235
    tissue_mask = np.logical_or(r,np.logical_or(g,b))
    return tissue_mask 
    

def TissueMaskGeneration_BIN(slide_obj, level):
    img_RGB = np.transpose(np.array(slide_obj.read_region((0, 0),
                       level,
                       slide_obj.level_dimensions[level]).convert('RGB')),
                       axes=[1, 0, 2])    
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
    img_S = img_HSV[:, :, 1]
    _,tissue_mask = cv2.threshold(img_S, 0, 255, cv2.THRESH_BINARY)
    return np.array(tissue_mask)

def TissueMaskGeneration_BIN_OTSU(slide_obj, level):
    img_RGB = np.transpose(np.array(slide_obj.read_region((0, 0),
                       level,
                       slide_obj.level_dimensions[level]).convert('RGB')),
                       axes=[1, 0, 2])    
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
    img_S = img_HSV[:, :, 1]
    _,tissue_mask = cv2.threshold(img_S, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return np.array(tissue_mask)

def labelthreshold(image, threshold=0.5):
    label = np.zeros_like(image)
    label[image >= threshold] = 1
    return label


def calc_jacc_score(x,y,smoothing=1):
    for var in [x,y]:
        np.place(var,var==255,1)
    
    numerator = np.sum(x*y)
    denominator = np.sum(np.logical_or(x,y))
    return (numerator+smoothing)/(denominator+smoothing)

# In[9]:

# In[10]:


def load_trained_models(model, path, patch_size=256):
    if model.__contains__('inception'):
        inception_model = get_inception_resnet_v2_unet_softmax((None, None), weights=None,)
        inception_model.load_weights(path)
        print ("Loaded Inception Model Weights")
        return inception_model
    elif model.__contains__('dense'):
        densenet_model = unet_densenet121((None, None), weights=None)
        densenet_model.load_weights(path)
        print ("Loaded densenet Model Weights")
        return densenet_model
    elif model.__contains__('deeplabv3'):        
        deeplab_model = Deeplabv3(input_shape=(patch_size, patch_size, 3), 
                          classes=2, 
                          backbone='xception', 
                          weights = 'pascal_voc',
                          OS=16, activation='softmax')
        deeplab_model.load_weights(path)
        print ("Loaded deeplabv3 Model Weights")
        return deeplab_model
    elif model == 'all':
        return [load_models(mdl) for mdl in ['inception', 'dense', 'deeplabv3']]


# In[11]:


# postprocessing helpers

def get_mean_img(probs, count_map):
   
    temp = []
    # dilate_kernel = np.ones((5, 5), dtype=np.uint8)
    for ii in probs:
        # ii = cv2.morphologyEx(np.array(ii), cv2.MORPH_DILATE, dilate_kernel).astype('float')
        temp.append(ii/count_map.astype('float'))
        
    probs = np.array(temp)
    mean_probs = np.mean(probs, axis=0)
    # mean_probs = cv2.morphologyEx(np.array(mean_probs), cv2.MORPH_ERODE, dilate_kernel)
    return mean_probs, np.mean(np.var(probs, axis=0))


def BinMorphoProcessMask(mask):
    """
    Binary operation performed on tissue mask
    """
    close_kernel = np.ones((50, 50), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((35, 35), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    return image_open 

iou = lambda x,y: 2.0*np.sum(x.astype("int")*y.astype("int"))/(np.sum(x.astype("int")+y.astype("int")) + 1e-3)


def apply_tta(imgs, tta):
    
    for i, img in enumerate(imgs):
        if tta == 'FLIP_LEFT_RIGHT':
            img = np.fliplr(img)
        elif tta == 'ROTATE_90':
            img = np.rot90(img)
        elif tta == 'ROTATE_180':
            img = np.rot90(img, 2)
        elif tta == 'ROTATE_270':
            img = np.rot90(img, 3)
        else:
            img = img
        imgs[i] = img
    return imgs



def transform_prob(data, tta):
    """
    Do inverse data augmentation
    """
    
    if tta == 'FLIP_LEFT_RIGHT':
        data = np.fliplr(data)
    elif tta == 'ROTATE_90':
        data = np.rot90(data, 3)
    elif tta == 'ROTATE_180':
        data = np.rot90(data, 2)
    elif tta == 'ROTATE_270':
        data = np.rot90(data, 1)
    else:
        data = data

    return data

def get_index(coord_ax, probs_map_shape_ax, grid_ax):
    """
    This function checks whether coordinates are within the WSI
    """
    # print (coord_ax, probs_map_shape_ax, grid_ax)
    _min = grid_ax//2
    _max = grid_ax//2

    ax_min = coord_ax - _min
    while ax_min < 0:
        _min -= 1
        ax_min += 1

    ax_max = coord_ax + _max
    while ax_max > probs_map_shape_ax:
        _max -= 1
        ax_max -= 1

    return _min, _max


# In[12]:

# Fully connected CRF post processing function
def do_crf(im, mask, n_labels, enable_color=False, zero_unsure=True):
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    # n_labels = len(set(labels.flat))
    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    if enable_color:
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
    Q = d.inference(5) # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    for u in unique_map: # get original labels back
        np.putmask(MAP, MAP == u, colors[u])
    return MAP

def post_process_crf(image, final_probabilities, num_cl):
    softmax = final_probabilities.squeeze()
    softmax = softmax.transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the unary_from_softmax for more information
    unary = unary_from_softmax(softmax, scale=None, clip=1e-5)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], num_cl)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                    img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    return res 


# In[30]:



