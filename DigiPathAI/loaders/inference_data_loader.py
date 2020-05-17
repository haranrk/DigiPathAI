from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import glob
import random
import time
import imgaug
from imgaug import augmenters as iaa
from PIL import Image
from tqdm import tqdm
import numpy as np 
from six.moves import range
import openslide
import tensorflow as tf
from torchvision import transforms  # noqa
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from helpers.utils import *



class WSIStridedPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """
    def __init__(self, wsi_path, mask_path, label_path=None, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE',                
                 level=5, sampling_stride=16, roi_masking=True):
        """
        Initialize the data producer.

        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format OR None
            label_mask_path: string, path to ground-truth label mask path in tif file or
                            None (incase of Normal WSI or test-time)
            image_size: int, size of the image before splitting into grid, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
            level: Level to extract the WSI tissue mask
            roi_masking: True: Multiplies the strided WSI with tissue mask to eliminate white spaces,
                                False: Ensures inference is done on the entire WSI   
            sampling_stride: Number of pixels to skip in the tissue mask, basically it's the overlap
                            fraction when patches are extracted from WSI during inference.
                            stride=1 -> consecutive pixels are utilized
                            stride= image_size/pow(2, level) -> non-overalaping patches 
        """
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._label_path = label_path
        self._image_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._level = level
        self._sampling_stride = sampling_stride
        self._roi_masking = roi_masking
        self._preprocess()

    def _preprocess(self):
        self._slide = openslide.OpenSlide(self._wsi_path)
        X_slide, Y_slide = self._slide.level_dimensions[0]
        factor = self._sampling_stride

        if self._label_path is not None:
            self._label_slide = openslide.OpenSlide(self._label_path)
        
        if self._mask_path is not None:
            mask_file_name = os.path.basename(self._mask_path)
            if mask_file_name.endswith('.npy'):
                self._mask = np.load(self._mask_path)
            if mask_file_name.endswith('.tif'):
                mask_obj = openslide.OpenSlide(self._mask_path)
                self._mask = np.array(mask_obj.read_region((0, 0),
                       self._level,
                       mask_obj.level_dimensions[self._level]).convert('L')).T
        else:
            # Generate tissue mask on the fly    
            self._mask = TissueMaskGeneration(self._slide, self._level)
           
        # morphological operations ensure the holes are filled in tissue mask
        # and minor points are aggregated to form a larger chunk         

        # self._mask = BinMorphoProcessMask(np.uint8(self._mask))
        # self._all_bbox_mask = get_all_bbox_masks(self._mask, factor)
        # self._largest_bbox_mask = find_largest_bbox(self._mask, factor)
        # self._all_strided_bbox_mask = get_all_bbox_masks_with_stride(self._mask, factor)

        X_mask, Y_mask = self._mask.shape
        # print (self._mask.shape, np.where(self._mask>0))
        # imshow(self._mask.T)
        # cm17 dataset had issues with images being power's of 2 precisely        
        if X_slide // X_mask != Y_slide // Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            ' X_slide / X_mask : {} / {},'
                            ' Y_slide / Y_mask : {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        self._resolution = np.round(X_slide * 1.0 / X_mask)
        if not np.log2(self._resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'
                            ' {}'.format(self._resolution))
             
        # all the idces for tissue region from the tissue mask  
        self._strided_mask =  np.ones_like(self._mask)
        ones_mask = np.zeros_like(self._mask)
        ones_mask[::factor, ::factor] = self._strided_mask[::factor, ::factor]
        if self._roi_masking:
            self._strided_mask = ones_mask*self._mask   
            # self._strided_mask = ones_mask*self._largest_bbox_mask   
            # self._strided_mask = ones_mask*self._all_bbox_mask 
            # self._strided_mask = self._all_strided_bbox_mask  
        else:
            self._strided_mask = ones_mask  
        # print (np.count_nonzero(self._strided_mask), np.count_nonzero(self._mask[::factor, ::factor]))
        # imshow(self._strided_mask.T, self._mask[::factor, ::factor].T)
        # imshow(self._mask.T, self._strided_mask.T)
 
        self._X_idcs, self._Y_idcs = np.where(self._strided_mask)        
        self._idcs_num = len(self._X_idcs)

    def __len__(self):        
        return self._idcs_num 

    def save_get_mask(self, save_path):
        np.save(save_path, self._mask)

    def get_mask(self):
        return self._mask

    def get_strided_mask(self):
        return self._strided_mask
    
    def __getitem__(self, idx):
        x_coord, y_coord = self._X_idcs[idx], self._Y_idcs[idx]

        # x = int(x_coord * self._resolution)
        # y = int(y_coord * self._resolution)    

        x = int(x_coord * self._resolution - self._image_size//2)
        y = int(y_coord * self._resolution - self._image_size//2)    

        img = self._slide.read_region(
            (x, y), 0, (self._image_size, self._image_size)).convert('RGB')
        
        if self._label_path is not None:
            label_img = self._label_slide.read_region(
                (x, y), 0, (self._image_size, self._image_size)).convert('L')
        else:
            label_img = Image.fromarray(np.zeros((self._image_size, self._image_size), dtype=np.uint8))
        
        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
            
        if self._rotate == 'ROTATE_90':
            img = img.transpose(Image.ROTATE_90)
            label_img = label_img.transpose(Image.ROTATE_90)
            
        if self._rotate == 'ROTATE_180':
            img = img.transpose(Image.ROTATE_180)
            label_img = label_img.transpose(Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(Image.ROTATE_270)
            label_img = label_img.transpose(Image.ROTATE_270)

        # PIL image:   H x W x C
        img = np.array(img, dtype=np.float32)
        label_img = np.array(label_img, dtype=np.uint8)

        if self._normalize:
            img = (img - 128.0)/128.0
   
        return (img, x_coord, y_coord, label_img)


if __name__ == '__main__':
    # Training Data Configuration    
    wsi_path = '/media/mak/mirlproject1/CAMELYON16/TrainingData/normal_tumor'
    label_path = '/media/mak/mirlproject1/CAMELYON16/TrainingData/lesion_masks'

    wsi_path_image = os.path.join(wsi_path, 'Tumor_001.tif')
    mask_path_image = None
    # label_path_image = os.path.join(label_path, 'Tumor_001_Mask.tif')
    label_path_image = None

    dataset_obj = WSIStridedPatchDataset(wsi_path_image, 
                                        mask_path_image,
                                        label_path_image,
                                        image_size=512,
                                        normalize=True,
                                        flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90',
                                        level=5, stride_factor=16, roi_masking=True)

    # for img, x_coord, y_coord, label_img in dataset_obj:
    #     print (img.dtype, label_img.dtype, x_coord, y_coord)
    #     imshow(img, label_img)
    #     # break

    dataloader = DataLoader(dataset_obj, batch_size=1, num_workers=0, drop_last=True)


    print (dataloader.dataset.__len__(), dataloader.__len__())
    i = 0
    start_time = time.time()
    # # imshow(dataset_obj.get_mask(), dataset_obj.get_strided_mask())
    for (data, x_mask, y_mask, label) in dataloader:
        image_patches = data.cpu().data.numpy()
        label_patches = label.cpu().data.numpy()
        x_coords = x_mask.cpu().data.numpy()
        y_coords = y_mask.cpu().data.numpy()
        print (image_patches.shape, image_patches.dtype, label_patches.shape, label_patches.dtype)
        # print (x_coords, y_coords)
        # For display 
        input_map = normalize_minmax(image_patches[0])
        label_map = label_patches[0]
        if np.sum(label_map) > 0:
            print (np.sum(label_map)/np.prod(label_map.shape))
            imshow(input_map, label_map)
            i += 1
            if i == 10:
                elapsed_time = time.time() - start_time
                print ("Elapsed Time", np.round(elapsed_time, decimals=2), "seconds")
                break