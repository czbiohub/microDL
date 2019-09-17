import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_path = r'\\flexo\MicroscopyData/ComputationalMicroscopy/Projects/virtualstaining/kidneyslice/' \
             '2019_02_15_kidney_slice/models_kidney_20190215'

metric_names = ['test_metric_xy', 'test_metric_xz', 'test_metric_3d']
tile_size_3d = 2048
metric_csv_names = ['{}_tile_size_3d_{}'.format(name, tile_size_3d) for name in metric_names]

output_path = r'\\flexo\MicroscopyData/ComputationalMicroscopy/Projects/virtualstaining/datastage_figures/kidneytissuep007/2D_model'
# corr_types = ['all', 'min_frac', 'min_frac_masked']
corr_types = ['all']
min_frac = 0.1
target_channel_names = ['F-actin', 'nuclei']

for metric_name, metric_csv_name in zip(metric_names, metric_csv_names):
    for target_channel_name in target_channel_names:
        metric_df = pd.read_csv(os.path.join(model_path, ''.join([metric_csv_name, '.csv'])),
                                index_col=0)
        for corr_type in corr_types:
            if corr_type == 'all':
                metric_df_sub = metric_df[metric_df['input channel name'].isin(
                                           ['retardance',
                                           'bright-field',
                                           'retardance + orientation x + orientation y + bright-field']) &
                                          (metric_df['target channel name'] == target_channel_name)]
                corr_col_name = 'pearson r'
            elif corr_type == 'min_frac':
                metric_df_sub = metric_df[(metric_df['foreground fraction'] > min_frac)]
                corr_col_name = 'pearson r'
            elif corr_type == 'min_frac_masked':
                metric_df_sub = metric_df[metric_df['foreground fraction'] > min_frac]
                corr_col_name = 'masked pearson r'
            else:
                ValueError('{} is not a valid correlation type'.format(corr_type))

            my_palette = {'F-actin': 'g', 'nuclei': 'm'}

            fig = plt.figure()
            fig.set_size_inches((6, 3))
            ax = sns.violinplot(x='input channel name', y=corr_col_name,
                                hue='target channel name',
                                data=metric_df_sub, scale='area',
                                linewidth=1, bw=0.05, inner='quartile',
                                split=False, palette=my_palette)
            ax.set_xticklabels(labels=['retardance',
                               'BF',
                               'retardance+slow axis+BF'])
            plt.xticks(rotation=25)
            # plt.title(''.join([metric_name, '_']))
            ax.set_ylim(bottom=0.5, top=1)
            # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
            # ax.legend(loc="lower right", borderaxespad=0.1)
            ax.get_legend().remove()
            plt.savefig(os.path.join(output_path, ''.join([metric_name, '_', corr_type, '_', target_channel_name, '.png'])),
                                     dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_path, ''.join([metric_name, '_', corr_type, '_', target_channel_name, '.pdf'])),
                                     dpi=300, bbox_inches='tight')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels(labels=[])

            plt.savefig(os.path.join(output_path, ''.join([metric_name, '_', corr_type, '_', target_channel_name, '_no_label', '.png'])),
                        dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_path, ''.join([metric_name, '_', corr_type, '_', target_channel_name, '_no_label', '.pdf'])),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)