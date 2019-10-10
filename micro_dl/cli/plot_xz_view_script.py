import os
import micro_dl.plotting.plot_utils as plot_utils

model_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
             '2019_02_15_kidney_slice/models_kidney_20190215'
model_dirs = [
        'kidney_3d_128_128_96_cyclr_2',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_ret_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_bf_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_bf_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_phase_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_phase_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm_v2',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_bf_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_phase_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_bf_actin_pix_iqr_norm',
        # 'Stack3_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
        # 'Stack7_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
        # 'target_actin',
        # 'target_nuclei',
        # 'input_retardance',
        # 'input_ret_ori',
        # 'input_phase',
        # 'input_BF',
        # '2D_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm'

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
# image_dir = '/CompMicro/Projects/virtualstaining/u2os_leonetti/2019_04_11_Ivan_Dividing_U2OS_v3/2019_04_11_HEK_U2OS_CollagenCoating/2019_04_11_U2OS_plin2_Col_Dividing2_2_2019_04_11_U2OS_plin2_Col_Dividing2_2/Pos0/'

# chan_names = ['Retardance', 'Orientation', 'Transmission', 'Polarization', 'phase3D']
# chan_names = ['phase3D']
# chan_name = 'c002'
chan_name = '*'
# t_idx = 35
# pos_idx = 0
# tol = 0.1
# plot_range = [612, 612, 800, 800]
t_idx = 0
pos_idx = 122
tol = 1
plot_range = [380, 0, 1200, 1200]
for model_dir in model_dirs:
# for chan_name in chan_names:
    image_dir = os.path.join(model_path, model_dir, 'predictions')
    # image_dir = '/data/anitha/models/kidney_3d_128_128_96_cyclr_2/predictions_0'
    # save_path = os.path.join(image_dir, 'figures', 'orthogonal_view_p{}.png'.format(pos_idx))
    save_path = os.path.join('/CompMicro/Projects/virtualstaining/datastage_figures/kidney_p122', 'orthogonal_view_{}_p{}.png'.format(model_dir, pos_idx))
    # save_path = os.path.join('/CompMicro/Projects/virtualstaining/datastage_figures/cells',
    #                          'xz_view_{}_p{}.png'.format(chan_name, pos_idx))
    plot_utils.save_center_slices(image_dir,
                                  t_idx,
                                   pos_idx,
                                   chan_name,
                                   save_path,
                                   plot_range=plot_range,
                                   mean_std=None,
                                   clip_limits=tol,
                                   margin=20,
                                   # z_scale=2.42,
                                    z_scale=1,
                                   channel_str=None,
                                   z_view=['xz','yz'],
                                   font_size=15,
                                   color_map='gray',
                                   fig_title=None)