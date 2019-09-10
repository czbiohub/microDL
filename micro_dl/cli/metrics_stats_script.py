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
        df_stats_tex_row['loss'] = 'L2 (Otsu)'
        df_stats = df_stats.append(df_stats_row[column_names], ignore_index=True)
        df_stats_tex = df_stats_tex.append(df_stats_tex_row[column_tex_names], ignore_index=True)
    df_stats.to_csv(os.path.join(model_path, ''.join(['metric_', target_chan_name, '.csv'])), sep=',')
    df_stats_tex.to_csv(os.path.join(model_path, ''.join(['metric_tex_', target_chan_name, '.csv'])),
                        sep='&', float_format='%.2f', index=False)


if __name__ == '__main__':
    # model_path = '/flexo/ComputationalMicroscopy/Projects/virtualstaining/kidneyslice/' \
    #              '2019_02_15_kidney_slice/models_kidney_20190215'
    # actin_model_dirs = [
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_trans_568_augmented_tf_20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ret_568_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ori_x_568_augmented_tf_20',
    #               '2D_tile256_step128_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ori_y_568_augmented_tf_20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_scat_568_augmented_tf_20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ret+trans_568_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ret+ori_x+ori_y_568_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ret+ori_x+ori_y+trans_568_augmented_tf20']
    # nuclei_model_dirs = [
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_trans_405_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ret_405_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ori_x_405_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ori_y_405_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_scat_405_augmented_tf_20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ret+trans_405_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ret+ori_x+ori_y_405_augmented_tf20',
    #               '2D_tile256_step128_registered_masked_clr_5e-5_6e-3_fltr16_256_do20_MSE_chan_ret+ori_x+ori_y+trans_405_augmented_tf20'
    #               ]
    # input_chan_names = ['bright-field',
    #                     'retardance',
    #                    'orientation x',
    #                    'orientation y',
    #                    'depolarization',
    #                    'retardance + bright-field',
    #                    'retardance + orientation x + orientation y',
    #                    'retardance + orientation x + orientation y + bright-field'
    #                    ]
    # translation_model = ['Slice->Slice']+[''] * (len(input_chan_names)-1)
    # input_chan_tex_names = ['BF',
    #                             '$\gamma$',
    #                             '$\phi_x$',
    #                             '$\phi_y$',
    #                             'DOP',
    #                             'BF,  $\gamma$',
    #                             '$\gamma$, $\phi_x$, $\phi_y$',
    #                             'BF, $\gamma$, $\phi_x$, $\phi_y$'
    #                             ]
    #
    #
    # input_image_path = '/flexo/ComputationalMicroscopy/Projects/virtualstaining/kidneyslice/' \
    #              '2019_02_15_kidney_slice/SMS_2018_1227_1433_1_BG_2019_0215_1337_1'
    # target_image_path = '/flexo/ComputationalMicroscopy/Projects/virtualstaining/kidneyslice/' \
    #                    '2019_02_15_kidney_slice/SMS_2018_1227_1433_1_BG_2019_0215_1337_1_registered'
    #
    # target_chan_names = ['nuclei', 'F-actin']
    # orientations = ['xy', 'xz', 'xyz']
    # metrics = ['corr', 'ssim']
    #
    # output_path = '/flexo/ComputationalMicroscopy/Projects/virtualstaining/datastage_figures/kidneytissuep007/2D_model'
    #
    # metrics_stats(model_path,
    #               actin_model_dirs,
    #               input_chan_names,
    #               input_chan_tex_names,
    #               'F-actin',
    #               metrics)
    #
    # metrics_stats(model_path,
    #               nuclei_model_dirs,
    #               input_chan_names,
    #               input_chan_tex_names,
    #               'nuclei',
    #               metrics)


    model_path = '/CompMicro/Projects/brainarchitecture/train_pool_H9_H78_GW20_GW24/models'

    model_dirs = ['pool_H9_H78_GW20_GW24_2D_MAE_phase_myelin_regis_pix_iqr_norm',
                  'pool_H9_H78_GW20_GW24_2D_MAE_4chan_phase_myelin_regis_pix_iqr_norm',
                  'pool_H9_H78_GW20_GW24_2D_MAE_4chan_bf_myelin_regis_pix_iqr_norm',
                  'pool_H9_H78_GW20_GW24_Stack_MAE_3chan_ret+ori_myelin_regis_pix_iqr_norm',
                  'pool_H9_H78_GW20_GW24_Stack_MAE_4chan_bf_myelin_regis_pix_iqr_norm',
                  ]

    input_chan_names = ['phase',
                        'retardance + orientation x + orientation y + phase',
                        'retardance + orientation x + orientation y + bright-field',
                        'retardance + orientation x + orientation y',
                        'retardance + orientation x + orientation y + bright-field',
                        ]
    translation_models = ['2D'] * 3 + ['2.5D'] * 2
    input_chan_tex_names = ['$\psi$',
                            '$\psi$, $\gamma$, $\phi_x$, $\phi_y$',
                            'BF, $\gamma$, $\phi_x$, $\phi_y$',
                            '$\gamma$, $\phi_x$, $\phi_y$',
                            'BF, $\gamma$, $\phi_x$, $\phi_y$',
                            ]

    orientations = ['xy']
    metrics = ['corr', 'ssim']

    output_path = model_path

    metrics_stats(model_path,
                  model_dirs,
                  input_chan_names,
                  input_chan_tex_names,
                  translation_models,
                  'myelin',
                  metrics)
