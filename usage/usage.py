import os
import time
import sys
import numpy as np
import openslide
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable



from PIL import Image
sys.path.append('..')
from DigiPathAI.Segmentation import getSegmentation

digestpath_imgs = ['../examples/colon-cancer-1-slide.tiff',
					'../examples/colon-cancer-2-slide.tiff']

paip_imgs       = ['']

camelyon_imgs   = ['']


for path in digestpath_imgs:
	ext = path[-4:]
	base_path = path[:-5]
	print (ext, base_path, base_path[:-5])
	getSegmentation(path, 
				patch_size  = 256, 
				stride_size = 128,
				batch_size  = 32,
				quick       = False,
				tta_list    = None, # ['FLIP_LEFT_RIGHT', 'ROTATE_90', 'ROTATE_180', 'ROTATE_270'],
				crf         = False,
				mask_path   = base_path + '-DigiPathAI_mask.' + ext,
				uncertainty_path   = base_path + '-DigiPathAI_uncertainty.'+ ext,
				status      = None,
				mode        = 'colon')

	slide = openslide.OpenSlide(path)
	level = len(slide.level_dimensions) - 1
	img_dimensions = slide.level_dimensions[-1]
	img = np.array(slide.read_region((0,0), level, img_dimensions).convert('RGB'))

	mask = openslide.OpenSlide(base_path + '-DigiPathAI_mask.' + ext)
	level = len(mask.level_dimensions) - 1
	dimensions = mask.level_dimensions[-1]
	mask = np.array(mask.read_region((0,0), level, dimensions).convert('L'))

	uncertainty = openslide.OpenSlide(base_path + '-DigiPathAI_uncertainty.'+ ext)
	level = len(uncertainty.level_dimensions) - 1
	dimensions = uncertainty.level_dimensions[-1]
	uncertainty = np.array(uncertainty.read_region((0,0), level, dimensions).convert('L'))

	mask = np.array(Image.fromarray(mask).resize(img_dimensions, Image.NEAREST))
	uncertainty = np.array(Image.fromarray(uncertainty).resize(img_dimensions))
	gt = np.array(Image.open(base_path[:-5] + 'gt.jpg').convert('RGB').resize(img_dimensions))

	plt.figure(figsize=(10, 40))
	gs = gridspec.GridSpec(1, 4)
	gs.update(wspace=0.02, hspace=0.02)

	ax = plt.subplot(gs[0, 0])
	im_ = ax.imshow(img)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_aspect('equal')
	ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
	

	ax = plt.subplot(gs[0, 1])
	im_ = ax.imshow(img)
	gt_ = ax.imshow(gt, alpha = 0.5)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_aspect('equal')
	ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
	

	ax = plt.subplot(gs[0, 2])
	im_ = ax.imshow(img)
	mask_ = ax.imshow(mask, alpha = 0.5)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_aspect('equal')
	ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
	
	ax = plt.subplot(gs[0, 3])
	im_ = ax.imshow(img)
	uncertain_ = ax.imshow(uncertainty, alpha = 0.5, cmap=plt.cm.RdBu_r)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(uncertain_, cax=cax)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_aspect('equal')
	ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )

	plt.show()