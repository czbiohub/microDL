
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import micro_dl.utils.aux_utils as aux_utils
np.seterr(all='ignore')


def sample_block_ceters(im, block_size):
    """Subdivide a 2D image in smaller blocks of size block_size and
    compute the median intensity value for each block. Any incomplete
    blocks (remainders of modulo operation) will be ignored.

    :param np.array im:         2D image
    :return np.array(float) sample_coords: Image coordinates for block
                                           centers
    :return np.array(float) sample_values: Median intensity values for
                                           blocks
    """

    im_shape = im.shape
    assert block_size < im_shape[0], "Block size larger than image height"
    assert block_size < im_shape[1], "Block size larger than image width"

    nbr_blocks_x = im_shape[0] // block_size
    nbr_blocks_y = im_shape[1] // block_size
    sample_coords = np.zeros((nbr_blocks_x * nbr_blocks_y, 2),
                             dtype=np.float64)
    sample_values = np.zeros((nbr_blocks_x * nbr_blocks_y,),
                             dtype=np.float64)
    for x in range(nbr_blocks_x):
        for y in range(nbr_blocks_y):
            idx = y * nbr_blocks_x + x
            sample_coords[idx, :] = [x * block_size + (block_size - 1) / 2,
                                     y * block_size + (block_size - 1) / 2]
            # get the center pixel value
            sample_values[idx] = im[1.5 * x * block_size, 1.5 * y * block_size]
    return sample_coords, sample_values

def distribution_plot(frames_metadata,
                      y_col,
                      output_path,
                      output_fname):
    my_palette = {'F-actin': 'g', 'nuclei': 'm'}

    fig = plt.figure()
    fig.set_size_inches((18, 9))
    ax = sns.violinplot(x='channel_name', y=y_col,
                        hue='dir_name',
                        bw=0.01,
                        data=frames_metadata, scale='area',
                        linewidth=1, inner='quartile',
                        split=False)
    # ax.set_xticklabels(labels=['retardance',
    #                            'BF',
    #                            'retardance+slow axis+BF'])
    plt.xticks(rotation=25)
    # plt.title(''.join([metric_name, '_']))
    # ax.set_ylim(bottom=0.5, top=1)
    # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    ax.legend(loc="upper left", borderaxespad=0.1)
    # ax.get_legend().remove()
    ax.set_ylabel('Mean intensity')
    plt.savefig(os.path.join(output_path, ''.join([output_fname, '.png'])),
                dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    # input_parent_dir = r'Y:\Projects\brainarchitecture'
    input_parent_dir = r'Y:\Projects\virtualstaining\kidneyslice\2019_02_15_kidney_slice'
    output_path = input_parent_dir
    # input_dirs = ['train_pool_H9_H78_GW20_GW24',
    #               r'2019_06_21_GW20point5_H30-135_594Fluoromyelin_10X_overlapping\SMS_20190621_1550_3_SMS_20190621_1550_3_fit_order2_registered',
    #               r'2019_06_20_GW24_reimaging_594Fluoromyelin_10X_test\SMS_062019_1250_1_SMS_062019_1250_1_fit_order2_registered']
    # input_dirs = ['train_pool_6_datasets']
    input_dirs = ['SMS_2018_1227_1433_1_SMS_2018_1227_1433_1_registered']
    normalize_im = 'dataset'
    min_fraction = 0.2
    frames_metadata = pd.DataFrame()
    ints_metadata = pd.DataFrame()
    pos_idx_cur = 0
    for input_dir in input_dirs:
        input_path = os.path.join(input_parent_dir, input_dir)
        frames_metadata_temp = aux_utils.read_meta(input_path)
        ints_metadata_temp = aux_utils.read_meta(input_path, 'ints_meta.csv')
        temp_pos_ids = frames_metadata_temp['pos_idx'].unique()
        temp_pos_ids.sort()
        pos_idx_map = dict(zip(temp_pos_ids, range(pos_idx_cur, pos_idx_cur + len(temp_pos_ids))))
        frames_metadata_temp['pos_idx'] = frames_metadata_temp['pos_idx'].map(pos_idx_map)
        frames_metadata = pd.concat([frames_metadata, frames_metadata_temp],
                                    axis=0, ignore_index=True)
        ints_metadata_temp['pos_idx'] = ints_metadata_temp['pos_idx'].map(pos_idx_map)
        ints_metadata = pd.concat([ints_metadata, ints_metadata_temp],
                                    axis=0, ignore_index=True)
        pos_idx_cur = pos_idx_map[temp_pos_ids[-1]] + 1

    time_ids = frames_metadata['time_idx'].unique()
    channel_ids = frames_metadata['channel_idx'].unique()
    pos_ids = frames_metadata['pos_idx'].unique()
    slice_ids = frames_metadata['slice_idx'].unique()
    dir_names = frames_metadata['dir_name'].unique()
    # dataset normalization is independent of slice ids
    slice_idx = 2
    pos_idx = 0
    ints_metadata['intensity norm'] = float('nan')
    for time_idx, channel_idx, dir_name in itertools.product(time_ids, channel_ids, dir_names):
        zscore_mean, zscore_std = aux_utils.get_zscore_params(
            time_idx=time_idx,
            channel_idx=channel_idx,
            slice_idx=slice_idx,
            pos_idx=pos_idx,
            dir_name=dir_name,
            depth=1,
            slice_ids=slice_ids,
            normalize_im=normalize_im,
            frames_metadata=ints_metadata,
            min_fraction=min_fraction
        )

        row_idxs = ((ints_metadata['time_idx'] == time_idx) &
                     (ints_metadata['channel_idx'] == channel_idx) &
                     (ints_metadata['dir_name'] == dir_name))
        ints_metadata.loc[row_idxs, 'zscore_mean'] = zscore_mean
        ints_metadata.loc[row_idxs, 'zscore_std'] = zscore_std
        row_idxs = row_idxs & (ints_metadata['fg_frac'] >= min_fraction)

        ints_metadata.loc[row_idxs, 'intensity norm'] = \
                (ints_metadata.loc[row_idxs, 'intensity'] - zscore_mean) / zscore_std

    distribution_plot(ints_metadata,
                      'intensity',
                      output_path,
                      'intensity_distribution_raw')

    distribution_plot(ints_metadata,
                      'intensity norm',
                      output_path,
                      'intensity_distribution_pix_iqr_norm')







