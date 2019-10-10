import os
from micro_dl.plotting import plot_tools

model_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
             '2019_02_15_kidney_slice/models_kidney_20190215'

actin_model_dirs = ['Stack_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
                    # 'Stack_fltr16_256_do20_otus_MAE_1chan_bf_actin_pix_iqr_norm',
                    'Stack_fltr16_256_do20_otus_MAE_1chan_phase_actin_pix_iqr_norm',
                    'Stack_fltr16_256_do20_otus_MAE_4chan_phase_actin_pix_iqr_norm',
                    # 'Stack_fltr16_256_do20_otus_MAE_4chan_bf_actin_pix_iqr_norm',
                    ]
nuclei_model_dirs = [
    'Stack_fltr16_256_do20_otus_MAE_1chan_ret_nuclei_pix_iqr_norm',
    # 'Stack_fltr16_256_do20_otus_MAE_1chan_bf_nuclei_pix_iqr_norm',
    'Stack_fltr16_256_do20_otus_MAE_1chan_phase_nuclei_pix_iqr_norm',
    'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm_v2',
    # 'Stack_fltr16_256_do20_otus_MAE_4chan_bf_nuclei_pix_iqr_norm',

]
input_chan_names = ['retardance',
                         # 'bright-field',
                         'phase',
                         '4_channel_phase',
                         # '4_channel_bright-field',
                         ]

target_chan_names = ['F-actin', 'nuclei']
output_path = '/CompMicro/Projects/virtualstaining/datastage_figures/kidney_p122/'
metric = 'ssim'
green_color = ['#3CB371', '#008000']
magenta_color = ['#EE82EE', '#FF00FF']
plot_colors = [green_color, magenta_color]
file_name = 'multi_contrast'

target_model_dirs = [actin_model_dirs, nuclei_model_dirs]

for target_chan, model_dirs, plot_color in zip(target_chan_names, target_model_dirs, plot_colors):
    dir_dict = {input_chan: os.path.join(model_path, model_dir, 'predictions')
                for model_dir, input_chan in zip(model_dirs, input_chan_names)}
    save_path = os.path.join(output_path,
                             ''.join([file_name, '_', metric, '_', target_chan, '.png']))

    plot_tools.singe_target_violins(dir_dict,
                                     save_path,
                                    metric,
                                     orientations=['xy', 'xz'],
                                     min_corr=0.5,
                                     plot_color=plot_color,
                                     use_labels=True)

    save_path = os.path.join(output_path,
                             ''.join([file_name, '_', metric, '_', target_chan, '_no_label', '.pdf']))

    plot_tools.singe_target_violins(dir_dict,
                                    save_path,
                                    metric,
                                    orientations=['xy', 'xz'],
                                    min_corr=0.5,
                                    plot_color=plot_color,
                                    use_labels=False)