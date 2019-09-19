"""
Run image_inference.py in parallel using multiprocessing and sub_process
"""
import os
from concurrent.futures import ProcessPoolExecutor
import subprocess
import shlex
import itertools

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
        # '2D_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm'
        # 'Stack_fltr16_256_do20_otus_MAE_3chan_ret+ori_actin_pix_iqr_norm',
        # 'Stack_fltr16_256_do20_otus_MAE_3chan_ret+ori_nuclei_pix_iqr_norm',
        # 'Stack3_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',
        'Stack_fltr16_256_do20_otus_MAE_4chan_phase_nuclei_pix_iqr_norm_v2',
        'Stack7_fltr16_256_do20_otus_MAE_1chan_ret_actin_pix_iqr_norm',

    ]
    image_path = '/CompMicro/Projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/SMS_2018_1227_1433_1_SMS_2018_1227_1433_1_registered'

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
    n_workers = 3
    gpu_ids = itertools.cycle(range(1, 4))
    for model_dir, gpu_id in zip(model_dirs, gpu_ids):
        argin = ''.join(['python cli/image_inference.py --model_dir ',
                         os.path.join(model_path, model_dir),
                         ' --save_to_model_dir ',
                         '--test_data ',
                         '--image_dir ', image_path,
                         ' --save_figs ',
                         '--ext .tif ',
                         '--gpu ', str(gpu_id),
                         ' --metrics ',
                         'coeff_determination ssim pearson_corr'])

        argin = shlex.split(argin)
        argin_list.append(argin)

    multiprocessing(sub_process, argin_list, n_workers=3)
