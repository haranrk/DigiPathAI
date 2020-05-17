import matplotlib.pyplot as plt
import cv2
import numpy as np

im_path = 'train_15/ref_Training_phase_1_003.png'
img = np.transpose(cv2.imread(im_path),[1,0,2])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
t_w = img.shape[0]
w = img.shape[0]//5
slide = img[4*w:t_w,:,:]
gt = img[3*w:4*w,:,:]
prob_map = img[1*w:2*w,:,:]
prob_map = np.mean(prob_map, axis=2)
prob_map/=255
pred = img[2*w:3*w,:,:]
# raise ValueError

fig, ax = plt.subplots(2, 2, figsize=(24, 24))
fig.tight_layout()
im_ = ax[0][0].imshow(slide)
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

pred_ = ax[1][1].imshow(pred,cmap='gray')
ax[1][1].set_xticklabels([])
ax[1][1].set_yticklabels([])
ax[1][1].set_xticks([])
ax[1][1].set_yticks([])
ax[1][1].set_aspect('equal')
ax[1][1].tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
# ax[1][1].title.set_text("Ground Truth")

prob_map_ = ax[1][0].imshow(prob_map, cmap=plt.cm.jet)
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

plt.savefig('characteristic-plots/im2.png',bbox_inches = 'tight',pad_inches = 0.1)
