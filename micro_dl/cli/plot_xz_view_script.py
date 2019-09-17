import os
import micro_dl.plotting.plot_utils as plot_utils

model_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
             '2019_02_15_kidney_slice/models_kidney_20190215'
model_dirs = [
        'Stack_fltr16_256_do20_otus_MAE_1chan_ret_nuclei_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_1chan_bf_actin_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_1chan_bf_nuclei_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_1chan_phase_nuclei_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_1chan_phase_actin_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_4chan_bf_nuclei_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_4chan_phase_actin_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_4chan_bf_actin_pix_iqr_norm',
        'target_actin',
        'target_nuclei',
        'input_retardance',
        '2D_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm'
            # '2D_fltr16_256_do20_otsu_MAE_ret_actin_augmented_tf_20',
            # '2D_fltr16_256_do20_otsu_MAE_ret_actin_augmented_tf_20_dataset_norm',
            # '2D_fltr16_256_do20_otsu_MSE_ret_actin_augmented_tf_20',
            # '2D_fltr16_256_do20_otsu_MSE_ret_actin_augmented_tf_20_dataset_norm',
            # '2D_fltr16_256_do20_otus_MAE_ret_actin_augmented_tf_20_volume_norm',
            # '2D_fltr16_256_do20_otus_masked_MAE_ret_actin_augmented_tf_20',
            # 'dsnorm_kidney_2019_z5_l1_nuclei',
            # 'dsnorm_kidney_2019_z5_l1_nuclei_4channels',
            # 'dsnorm_kidney_2019_z5_l1_rosin',
            # 'dsnorm_kidney_2019_z5_l1_rosin_4channels',
              ]
pos_idx = 122
tol = 1
plot_range = [380, 0, 1200, 1200]
for model_dir in model_dirs:
    image_dir = os.path.join(model_path, model_dir, 'predictions')
    save_path = os.path.join(image_dir, 'figures', 'orthogonal_view_p{}.png'.format(pos_idx))
    plot_utils.save_center_slices(image_dir,
                           pos_idx,
                           save_path,
                           plot_range=plot_range,
                           mean_std=None,
                           clip_limits=tol,
                           margin=20,
                           z_scale=2,
                           channel_str=None,
                           font_size=15,
                           color_map='gray',
                           fig_title=None)