import os
import time
import glob
import sys
import numpy as np
import openslide
import matplotlib.pyplot as plt
import cv2
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable



from PIL import Image
sys.path.append('..')
from DigiPathAI.Segmentation import getSegmentation


digestpath_imgs = ['../examples/colon-cancer-1.tiff']

models = ['dense']#, 'inception', 'deeplabv3', 'ensemble', 'epistemic']


for path in digestpath_imgs:
  ext = os.path.splitext(path)[1]
  base_path = os.path.splitext(path)[0]

  print (ext, base_path, base_path[:-5])
  quick = True
  tta_list = ['FLIP_LEFT_RIGHT', 'ROTATE_90'] #, 'ROTATE_180', 'ROTATE_270']
  for model in models:

    print (model, quick, path, "======================================")
    if model == 'ensemble': 
      quick = False
    elif model == 'epistemic':
      quick = False
      tta_list = None
    """
    getSegmentation(path, 
          patch_size  = 256, 
          stride_size = 128,
          batch_size  = 4,
          quick       = quick,
          tta_list    = tta_list,
          crf         = False,
          probs_path  = base_path + '-DigiPathAI_{}_probs'.format(model) + '.tiff',
          mask_path   = base_path + '-DigiPathAI_{}_mask'.format(model) + '.tiff',
          uncertainty_path   = base_path + '-DigiPathAI_{}_uncertainty'.format(model)+ '.tiff',
          status      = None,
          mask_level = 4,
          model       = model,
          mode        = 'colon')
    """
    slide = openslide.OpenSlide(path)
    level = len(slide.level_dimensions) - 1
    img_dimensions = slide.level_dimensions[-1]
    img = np.array(slide.read_region((0,0), level, img_dimensions).convert('RGB'))

    mask = openslide.OpenSlide(base_path + '-DigiPathAI_{}_mask'.format(model) + '.tiff')
    level = np.where([1 if ((dim[0] == img_dimensions[0])*(dim[1] == img_dimensions[1])) else 0 for dim in mask.level_dimensions])[0]
    mask = np.array(mask.read_region((0,0), level, img_dimensions).convert('L'))
    
    probs = openslide.OpenSlide(base_path + '-DigiPathAI_{}_probs'.format(model) + '.tiff')
    level = np.where([1 if ((dim[0] == img_dimensions[0])*(dim[1] == img_dimensions[1])) else 0 for dim in probs.level_dimensions])[0]
    probs = np.array(probs.read_region((0,0), level, img_dimensions).convert('L'))/255.0

    gt = np.array(Image.open(base_path+ 'gt.jpg').convert('L').resize(img_dimensions))

    fig, ax = plt.subplots(2, 2, figsize=(14, 20))
    fig.tight_layout()
    im_ = ax[0][0].imshow(img)
    ax[0][0].set_xticklabels([])
    ax[0][0].set_yticklabels([])
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])
    ax[0][0].set_aspect('equal')
    ax[0][0].tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
    # ax[0][0].title.set_text("WSI Slide")

    gt_ = ax[0][1].imshow(gt,cmap='gray')
    ax[0][1].set_xticklabels([])
    ax[0][1].set_yticklabels([])
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])
    ax[0][1].set_aspect('equal')
    ax[0][1].tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
    # ax[0][1].title.set_text("Ground Truth")

    pred_ = ax[1][1].imshow(mask,cmap='gray')
    ax[1][1].set_xticklabels([])
    ax[1][1].set_yticklabels([])
    ax[1][1].set_xticks([])
    ax[1][1].set_yticks([])
    ax[1][1].set_aspect('equal')
    ax[1][1].tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
    # ax[1][1].title.set_text("Ground Truth")

    prob_map_ = ax[1][0].imshow(probs, cmap=plt.cm.jet)
    ax[1][0].set_xticklabels([])
    ax[1][0].set_yticklabels([])
    ax[1][0].set_xticks([])
    ax[1][0].set_yticks([])
    ax[1][0].set_aspect('equal')
    ax[1][0].tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
    # ax[1][0].title.set_text("Probability Map")

    cax = fig.add_axes([ax[1][0].get_position().x1 + 0.01,
              ax[1][0].get_position().y0,
              0.01,
              ax[1][0].get_position().y1-ax[1][0].get_position().y0])
    fig.colorbar(prob_map_, cax=cax)

    plt.savefig('im2.png',bbox_inches = 'tight',pad_inches = 0.1)
