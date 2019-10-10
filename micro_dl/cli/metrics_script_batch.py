"""
Run metrics_script.py in parallel using multiprocessing and sub_process
"""
import os
from concurrent.futures import ProcessPoolExecutor
import subprocess
import shlex

def multiprocessing(func, args, n_workers = 8):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        executor.map(func, args)

def sub_process(argin):
    p = subprocess.Popen(argin)
    p.wait()
    print(p.stdout)

if __name__ == '__main__':
    model_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
                 '2019_02_15_kidney_slice/models_kidney_20190215'
    model_dirs = [
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_ret_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_bf_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_bf_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_phase_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_1chan_phase_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_bf_nuclei_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_phase_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_bf_actin_pix_iqr_norm',
        '2D_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm_v2',
        # 'kidney_3d_128_128_96_cyclr_2',
        # 'Stack_fltr16_256_do20_otus_MAE_3chan_ret+ori_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_3chan_ret+ori_nuclei_pix_iqr_norm',
        'Stack3_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm_v2',
        # 'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm_v2',
        'Stack7_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm_v2',
    ]
    image_path = '/CompMicro/Projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/SMS_2018_1227_1433_1_SMS_2018_1227_1433_1_registered'

    # model_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
    #              '2019_02_15_kidney_slice/models_kidney_20190215'
    # model_dirs = [
    #             '2D_fltr16_256_do20_otsu_MAE_ret_actin_augmented_tf_20',
    #             '2D_fltr16_256_do20_otsu_MAE_ret_actin_augmented_tf_20_dataset_norm',
    #             '2D_fltr16_256_do20_otsu_MSE_ret_actin_augmented_tf_20',
    #             '2D_fltr16_256_do20_otsu_MSE_ret_actin_augmented_tf_20_dataset_norm',
    #             '2D_fltr16_256_do20_otus_MAE_ret_actin_augmented_tf_20_volume_norm',
    #             '2D_fltr16_256_do20_otus_masked_MAE_ret_actin_augmented_tf_20',
    #             'dsnorm_kidney_2019_z5_l1_nuclei',
    #             'dsnorm_kidney_2019_z5_l1_nuclei_4channels',
    #             'dsnorm_kidney_2019_z5_l1_rosin',
    #             'dsnorm_kidney_2019_z5_l1_rosin_4channels',
    #               ]
    # image_path = '/CompMicro/Projects/virtualstaining/kidneyslice/' \
    #              '2019_02_15_kidney_slice/SMS_2018_1227_1433_1_BG_2019_0215_1337_1_registered/'

    # model_path = '/CompMicro/Projects/brainarchitecture/train_pool_H9_H78_GW20_GW24/models'
    #
    # model_dirs = ['pool_H9_H78_GW20_GW24_2D_MAE_phase_myelin_regis_pix_iqr_norm',
    #               'pool_H9_H78_GW20_GW24_Stack_MAE_3chan_ret+ori_myelin_regis_pix_iqr_norm',
    #               'pool_H9_H78_GW20_GW24_Stack_MAE_4chan_bf_myelin_regis_pix_iqr_norm',
    #               'pool_H9_H78_GW20_GW24_2D_MAE_4chan_bf_myelin_regis_pix_iqr_norm',
    #               'pool_H9_H78_GW20_GW24_2D_MAE_4chan_phase_myelin_regis_pix_iqr_norm',
    #               ]
    # image_path = '/CompMicro/Projects/brainarchitecture/train_pool_H9_H78_GW20_GW24'
    argin_list = []
    for model_dir in model_dirs:
        argin = ''.join(['python cli/metrics_script.py --model_dir ',
                         os.path.join(model_path, model_dir),
                         ' --test_data ',
                         '--image_dir ',
                         image_path,
                         ' --metrics ', 'ssim corr r2',
                         ' --orientations ', 'xy xz xyz',
                         ' --ext ', '.tif'])
        argin = shlex.split(argin)
        argin_list.append(argin)

    multiprocessing(sub_process, argin_list, n_workers=11)
