import os
import numpy as np
import yaml
import cv2


def im_bit_convert(im, bit=16, norm=False, limit=[]):
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm:
        if not limit:
            limit = [np.nanmin(im[:]), np.nanmax(im[:])] # scale each image individually based on its min and max
        im = (im-limit[0])/(limit[1]-limit[0])*(2**bit-1)
    im = np.clip(im, 0, 2**bit-1) # clip the values to avoid wrap-around by np.astype
    if bit==8:
        im = im.astype(np.uint8, copy=False) # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False) # convert to 16 bit
    return im


def im_adjust(img, tol=1):
    """
    Adjust contrast of the image

    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=8, norm=True, limit=limit.tolist())
    return im_adjusted

def read_img(image_path, tiff_file_name, tol=1):
    """
    Read image and adjust contrast

    """
    tiff_file_path = os.path.join(image_path, tiff_file_name)
    # img = cv2.imread(tiff_file_path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(tiff_file_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError('No such file {}'.format(tiff_file_path))
    img = im_adjust(img, tol=tol)
    return img

def export_img(im_input, output_path, tiff_file_name):
    if len(im_input.shape) < 3:
        cv2.imwrite(os.path.join(output_path, tiff_file_name), im_input)
    else:
        cv2.imwrite(os.path.join(output_path, tiff_file_name),
                    cv2.cvtColor(im_input, cv2.COLOR_RGB2BGR))

def generate_slice_png(pred_img_dir,
                       actin_model_dirs,
                       nuclei_model_dirs,
                       input_chan_names,
                       target_chan_names,
                       pred_input_chan_names,
                       pred_target_chan_names,
                       input_image_path,
                       target_image_path,
                       output_path,
                       t_idx,
                       pos_idx,
                       z_idx,
                       tol,
                       plot_range=None):

    # os.makedirs(output_path, exist_ok=True)
    # for chan_name in input_chan_names:
    #     tiff_file_name = 'img_' + chan_name + '_t%03d_p%03d_z%03d.tif' % (t_idx, pos_idx, z_idx)
    #     im_input = read_img(input_image_path, tiff_file_name, tol=tol)
    #     if plot_range is not None:
    #         im_input = im_input[plot_range[0]:plot_range[0] + plot_range[2],
    #                   plot_range[1]:plot_range[1] + plot_range[3], ...]
    #     png_file_name = ''.join([tiff_file_name[:-4], '.png'])
    #     export_img(im_input, output_path, png_file_name)

    # im_target_2chan = []
    # for chan_name in target_chan_names:
    #     tiff_file_name = 'img_' + chan_name + '_t%03d_p%03d_z%03d.tif' % (t_idx, pos_idx, z_idx)
    #     im_target = read_img(target_image_path, tiff_file_name, tol=tol)
    #     im_target_2chan.append(im_target)
    #     tiff_file_name = ''.join([tiff_file_name[:-4], '.png'])
    #     export_img(im_target, output_path, tiff_file_name)
    # # make green-magenta overlay
    # im_target_2chan.append(im_target_2chan[0])
    # im_target_2chan = np.stack(im_target_2chan, -1)
    #
    # tiff_file_name = 'img_{}_{}_t{:03d}_p{:03d}_z{:03d}.png'.format(
    #     target_chan_names[0], target_chan_names[1], t_idx, pos_idx, z_idx)
    # export_img(im_target_2chan, output_path, tiff_file_name)


    for actin_model_dir, nuclei_model_dir, input_channel_name in \
            zip(actin_model_dirs, nuclei_model_dirs, pred_input_chan_names):
        print('processing {} & {}...'.format(actin_model_dir, nuclei_model_dir))
        # config_name = os.path.join(pred_img_dir, actin_model_dir, 'config.yml')
        # with open(config_name, 'r') as f:
        #     config = yaml.load(f)
        # dataset_config = config['dataset']
        # input_channel = dataset_config['input_channels']
        # target_channel = dataset_config['target_channels'][0]
        # if len(input_channel) == 1 and isinstance(input_channel, list):
        #     input_channel = input_channel[0]
        # chan_name = ''.join(["c" + str(input_channel).zfill(3)])
        im_pred_2chan = []
        for model_dir, pred_target_chan_name in zip([nuclei_model_dir, actin_model_dir], pred_target_chan_names):

            tiff_file_name = 'orthogonal_view_{}_p{}.png'.format(model_dir, pos_idx)
            im_pred = read_img(pred_img_dir, tiff_file_name, tol=tol)
            im_pred_2chan.append(im_pred)
            # tiff_file_name = 'img_{}_to_{}_t{:03d}_p{:03d}_z{:03d}.png'.format(
            #     input_channel_name, pred_target_chan_name, t_idx, pos_idx, z_idx)
            # export_img(im_pred, output_path, tiff_file_name)
            # make green-magenta overlay
        im_pred_2chan.append(im_pred_2chan[0])
        im_pred_2chan = np.stack(im_pred_2chan, -1)
        print(im_pred.shape)
        print(im_pred_2chan.shape)
        print(im_pred_2chan.dtype)
        tiff_file_name = 'img_{}_to_{}_{}_p{:03d}.png'.format(
            input_channel_name, pred_target_chan_names[0], pred_target_chan_names[1], pos_idx)
        export_img(im_pred_2chan, output_path, tiff_file_name)



if __name__ == '__main__':
    # pred_img_dir = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
    #              '2019_02_15_kidney_slice/models_kidney_20190215'

    pred_img_dir = '/CompMicro/Projects/virtualstaining/datastage_figures/kidney_p122'
                             
    actin_model_dirs = ['Stack_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
                        'Stack_fltr16_256_do20_otus_MAE_1chan_bf_actin_pix_iqr_norm',
                        'Stack_fltr16_256_do20_otus_MAE_1chan_phase_actin_pix_iqr_norm',
                        'Stack_fltr16_256_do20_otus_MAE_4chan_phase_actin_pix_iqr_norm',
                        'Stack_fltr16_256_do20_otus_MAE_4chan_bf_actin_pix_iqr_norm',
                        'target_actin',
                        ]
    nuclei_model_dirs = [
                   'Stack_fltr16_256_do20_otus_MAE_1chan_ret_nuclei_pix_iqr_norm',
                    'Stack_fltr16_256_do20_otus_MAE_1chan_bf_nuclei_pix_iqr_norm',
                    'Stack_fltr16_256_do20_otus_MAE_1chan_phase_nuclei_pix_iqr_norm',
                    'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm_v2',
                    'Stack_fltr16_256_do20_otus_MAE_4chan_bf_nuclei_pix_iqr_norm',
                    'target_nuclei',
                  ]
    pred_input_chan_names = ['retardance',
                           # 'orientation x',
                           # 'orientation y',
                           'bright-field',
                           'phase',
                           # 'retardance + bright-field',
                           # 'retardance + orientation x + orientation y',
                           '4_channel_phase',
                           '4_channel_bright-field',
                           'target',
                           ]

    input_image_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
                 '2019_02_15_kidney_slice/SMS_2018_1227_1433_1_SMS_2018_1227_1433_1/'
    # input_image_path = '/CompMicro/Projects/virtualstaining/u2os_leonetti/2019_04_11_Ivan_Dividing_U2OS_v3/2019_04_11_HEK_U2OS_CollagenCoating/2019_04_11_U2OS_plin2_Col_Dividing2_2_2019_04_11_U2OS_plin2_Col_Dividing2_2/Pos0/'
    target_image_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
                       '2019_02_15_kidney_slice/SMS_2018_1227_1433_1_BG_2019_0215_1337_1_registered'
    input_chan_names = ['Retardance', 'Orientation', 'Brightfield_computed', 'Polarization', 'phase',
                       'Retardance+Orientation']
    # input_chan_names = ['Retardance', 'Orientation', 'Transmission', 'Polarization', 'phase3D',
    #                     'Retardance+Orientation']
    target_chan_names = ['405', '568']
    pred_target_chan_names = ['nuclei', 'F-actin']
    # output_path = '/CompMicro/Projects/virtualstaining/datastage_figures/cells/'
    output_path = '/CompMicro/Projects/virtualstaining/datastage_figures/kidney_p122/'
    tIdx = 0
    posIdx = 122
    zIdx = 22
    tol = 1

    # tIdx = 35
    # posIdx = 0
    # zIdx = 9
    # tol = 0.1
    plot_range = [380, 0, 1200, 1200]
    # plot_range = [612, 612, 800, 800]
    generate_slice_png(pred_img_dir,
                       actin_model_dirs,
                       nuclei_model_dirs,
                       input_chan_names,
                       target_chan_names,
                       pred_input_chan_names,
                       pred_target_chan_names,
                       input_image_path,
                       target_image_path,
                       output_path,
                       tIdx,
                       posIdx,
                       zIdx,
                       tol,
                       plot_range,
                       )