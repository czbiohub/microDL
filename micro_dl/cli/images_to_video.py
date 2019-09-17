import cv2
import numpy as np
import os
import natsort
import glob

image_dir = '/flexo/ComputationalMicroscopy/Projects/virtualstaining/u2os_leonetti/2019_04_11_Ivan_Dividing_U2OS_v3/2019_04_11_HEK_U2OS_CollagenCoating/2019_04_11_U2OS_plin2_Col_Dividing2_2_2019_04_11_U2OS_plin2_Col_Dividing2_2/Pos0/'
save_dir = '/flexo/ComputationalMicroscopy/Projects/virtualstaining/u2os_leonetti/2019_04_11_Ivan_Dividing_U2OS_v3/movie_frames/'
pos_idx = 0
min_percentile = 2
max_percentile = 99.9
channel_str = 'Retardance+Orientation_t035'
search_str = os.path.join(image_dir, "*p{:03d}*".format(pos_idx))
slice_names = natsort.natsorted(glob.glob(search_str))
save_dir = os.path.join(save_dir, channel_str)
os.makedirs(save_dir, exist_ok=True)

if channel_str is not None:
    slice_names = [s for s in slice_names if channel_str in s]
im_stack = []
for im_z in slice_names:
    im_stack.append(cv2.imread(im_z, cv2.IMREAD_ANYDEPTH))
im_stack = np.stack(im_stack, axis=-1)
im_stack = im_stack[559:1461, 579:1481, :]

pmin, pmax = np.percentile(im_stack, [min_percentile, max_percentile])
im_norm = np.clip(im_stack, pmin, pmax).astype(np.float32)

im_norm = ((im_norm - im_norm.min()) / (im_norm.max() - im_norm.min())) * 255
im_norm = im_norm.astype(np.uint8)

for i in range(im_norm.shape[2]):
    file_name = os.path.join(save_dir, 'im_{:02d}.png'.format(i))
    cv2.imwrite(file_name, im_norm[..., i])
