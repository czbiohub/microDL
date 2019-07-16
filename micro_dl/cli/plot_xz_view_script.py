import os
import micro_dl.plotting.plot_utils as plot_utils
model_path = '/flexo/ComputationalMicroscopy/Projects/virtualstaining/kidneyslice/' \
                 '2019_02_15_kidney_slice/models_kidney_20190215'

model_dirs = ['2D_fltr16_256_do20_otsu_MAE_ret_actin_augmented_tf_20',
              '2D_fltr16_256_do20_otsu_MAE_ret_actin_augmented_tf_20_dataset_norm',
              '2D_fltr16_256_do20_otus_MAE_ret_actin_augmented_tf_20_volume_norm']

pos_idx = 122
tol = 1
for model_dir in model_dirs:
    image_dir = os.path.join(model_path, model_dir, 'predictions')
    save_path = os.path.join(image_dir, 'figures', 'orthogonal_view_p{}.jpg'.format(pos_idx))
    plot_utils.save_center_slices(image_dir,
                           pos_idx,
                           save_path,
                           mean_std=None,
                           clip_limits=tol,
                           margin=20,
                           z_scale=5,
                           channel_str=None,
                           font_size=15,
                           color_map='gray',
                           fig_title=None)