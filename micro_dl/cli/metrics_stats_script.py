import os
import pandas as pd
from itertools import product


def metrics_stats(model_path,
                  model_dirs,
                  input_chan_names,
                  input_chan_tex_names,
                  translation_models,
                  target_chan_name,
                  metrics):
    """
    Read the metrics.csv from a list model directories, compute median of metrics
    and export medians in CSV and TEX format
    """

    column_names = ['{}_{}'.format(metric, orientation)
                  for metric, orientation in product(metrics, orientations)]
    column_tex_names = ['input channels', 'loss'] + column_names
    column_names = ['input channels', 'model'] + column_names

    df_stats = pd.DataFrame(
        columns=column_names
    )
    df_stats_tex = pd.DataFrame(
        columns=column_tex_names
    )
    for model_dir, input_chan_name, input_chan_tex_name, model in \
            zip(model_dirs, input_chan_names, input_chan_tex_names, translation_models):
        print('processing {}...'.format(model_dir))
        df_stats_row = []
        for orientation in orientations:
            metric_csv_name = 'metrics_{}.csv'.format(orientation)
            metric_df = pd.read_csv(os.path.join(model_path, model_dir, 'predictions', metric_csv_name),
                                    index_col=0)
            metric_column_names = ['{}_{}'.format(metric, orientation) for metric in metrics]
            metric_df_stats = metric_df[metrics].agg('median')
            metric_df_stats = metric_df_stats.to_frame().T
            metric_df_stats.columns = metric_column_names
            df_stats_row.append(metric_df_stats)
        df_stats_row = pd.concat(df_stats_row, axis=1)
        df_stats_tex_row = df_stats_row.copy()
        df_stats_row['input channels'] = input_chan_name
        df_stats_row['model'] = model
        df_stats_tex_row['input channels'] = input_chan_tex_name
        df_stats_tex_row['loss'] = 'L1'
        df_stats = df_stats.append(df_stats_row[column_names], ignore_index=True)
        df_stats_tex = df_stats_tex.append(df_stats_tex_row[column_tex_names], ignore_index=True)
    df_stats.to_csv(os.path.join(model_path, ''.join(['metric_', target_chan_name, '.csv'])), sep=',')
    df_stats_tex.to_csv(os.path.join(model_path, ''.join(['metric_tex_', target_chan_name, '.csv'])),
                        sep='&', float_format='%.2f', index=False)


if __name__ == '__main__':
    model_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
                 '2019_02_15_kidney_slice/models_kidney_20190215'
    actin_model_dirs = ['2D_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm_v3',
                        'Stack3_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm_v3',
                        'Stack_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
                        'Stack7_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm_v3',
                        'kidney_3d_128_128_96_cyclr_2',
                        ]
    # actin_model_dirs = ['Stack_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
    #                     'Stack_fltr16_256_do20_otus_MAE_1chan_bf_actin_pix_iqr_norm',
    #                     'Stack_fltr16_256_do20_otus_MAE_1chan_phase_actin_pix_iqr_norm',
    #                     'Stack_fltr16_256_do20_otus_MAE_4chan_phase_actin_pix_iqr_norm',
    #                     'Stack_fltr16_256_do20_otus_MAE_4chan_bf_actin_pix_iqr_norm',
    #                     ]
    nuclei_model_dirs = [
                   'Stack_fltr16_256_do20_otus_MAE_1chan_ret_nuclei_pix_iqr_norm',
                    'Stack_fltr16_256_do20_otus_MAE_1chan_bf_nuclei_pix_iqr_norm',
                    'Stack_fltr16_256_do20_otus_MAE_1chan_phase_nuclei_pix_iqr_norm',
                    'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm_v2',
                    'Stack_fltr16_256_do20_otus_MAE_4chan_bf_nuclei_pix_iqr_norm',
                  ]

    # input_chan_names = ['retardance',
    #                          # 'orientation x',
    #                          # 'orientation y',
    #                          'bright-field',
    #                          'phase',
    #                          # 'retardance + bright-field',
    #                          # 'retardance + orientation x + orientation y',
    #                          '4_channel_phase',
    #                          '4_channel_bright-field',
    #                          ]
    input_chan_names = ['retardance'] * len(actin_model_dirs)

    # translation_models = ['2.5D']+[''] * (len(input_chan_names)-1)
    translation_models = ['2D', '2.5D, z=3', '2.5D, z=5', '2.5D, z=7', '3D']


    input_chan_tex_names = [r'$\rho$',
                            '$\phi$',
                            'BF',
                            r'$\phi$, $\rho$, $\omega_x$, $\omega_y$',
                            r'BF, $\rho$, $\omega_x$, $\omega_y$',
                            ]
    # input_chan_tex_names = [r'$\rho$'] * len(actin_model_dirs)

    target_chan_names = ['nuclei', 'F-actin']
    orientations = ['xy', 'xz', 'xyz']
    metrics = ['corr', 'ssim']

    # output_path = '/CompMicro/Projects/virtualstaining/datastage_figures/kidney_p122'

    metrics_stats(model_path,
                  actin_model_dirs,
                  input_chan_names,
                  input_chan_tex_names,
                  translation_models,
                  'F-actin',
                  metrics)

    # metrics_stats(model_path,
    #               nuclei_model_dirs,
    #               input_chan_names,
    #               input_chan_tex_names,
    #               translation_models,
    #               'nuclei',
    #               metrics)


    # model_path = '/CompMicro/Projects/brainarchitecture/train_pool_H9_H78_GW20_GW24/models'
    #
    # model_dirs = ['pool_H9_H78_GW20_GW24_2D_MAE_phase_myelin_regis_pix_iqr_norm',
    #               'pool_H9_H78_GW20_GW24_2D_MAE_4chan_phase_myelin_regis_pix_iqr_norm',
    #               'pool_H9_H78_GW20_GW24_2D_MAE_4chan_bf_myelin_regis_pix_iqr_norm',
    #               'pool_H9_H78_GW20_GW24_Stack_MAE_3chan_ret+ori_myelin_regis_pix_iqr_norm',
    #               'pool_H9_H78_GW20_GW24_Stack_MAE_4chan_bf_myelin_regis_pix_iqr_norm',
    #               ]
    #
    # input_chan_names = ['phase',
    #                     'retardance + orientation x + orientation y + phase',
    #                     'retardance + orientation x + orientation y + bright-field',
    #                     'retardance + orientation x + orientation y',
    #                     'retardance + orientation x + orientation y + bright-field',
    #                     ]
    # translation_models = ['2D'] * 3 + ['2.5D'] * 2
    # input_chan_tex_names = ['$\psi$',
    #                         '$\psi$, $\gamma$, $\phi_x$, $\phi_y$',
    #                         'BF, $\gamma$, $\phi_x$, $\phi_y$',
    #                         '$\gamma$, $\phi_x$, $\phi_y$',
    #                         'BF, $\gamma$, $\phi_x$, $\phi_y$',
    #                         ]
    #
    # orientations = ['xy']
    # metrics = ['corr', 'ssim']
    #
    # output_path = model_path
    #
    # metrics_stats(model_path,
    #               model_dirs,
    #               input_chan_names,
    #               input_chan_tex_names,
    #               translation_models,
    #               'myelin',
    #               metrics)
