"""Evaulate performance of reg model"""
import keras.backend as K
from keras import Model
import numpy as np
import os
import pickle
import yaml

from micro_dl.input.dataset_regression import RegressionDataSet
from micro_dl.train.model_inference import load_model
import micro_dl.utils.train_utils as train_utils 


def coeff_of_deter_per_index(y_true, y_pred):
    """Goodness of fit"""
    
    ss_res = np.sum(np.square(y_true - y_pred), axis=0)
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)), axis=0)
    return ss_res, ss_tot #( 1 - ss_res/(ss_tot + K.epsilon()) )

def eval_abs_deviation(y_true, y_pred):
    """Absolute deviation"""
    
    abs_dev = np.abs(y_true - y_pred)
    min_dev = np.min(abs_dev, axis=0)
    max_dev = np.max(abs_dev, axis=0)
    mean_dev = np.mean(abs_dev, axis=0)
    return min_dev, max_dev, mean_dev

def eval_per_abs_deviation(y_true, y_pred):
    """Absolute deviation"""
    
    per_dev = np.abs(y_true - y_pred) / (np.abs(y_true) + K.epsilon())
    min_dev = np.min(per_dev, axis=0)
    max_dev = np.max(per_dev, axis=0)
    mean_dev = np.mean(per_dev, axis=0)
    return min_dev, max_dev, mean_dev
    
def predict_coef(model, input_fnames, num_focal_planes, batch_size, num_reg_coeff):
    """Predict zernike coeff"""
    
    reg_ds = RegressionDataSet(input_fnames, num_focal_planes, num_reg_coeff, batch_size)
    num_batches = reg_ds.__len__()
    target_zer_list = []
    pred_zer_list = []
    
    batch_ss_res = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_ss_tot = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_min_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_max_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_mean_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_per_min_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_per_max_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_per_mean_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    
    for batch_idx in range(num_batches):
        input_batch, target_batch = reg_ds.__getitem__(batch_idx)
        target_zer_list.append(target_batch)
        pred_batch = model.predict(input_batch)
        pred_zer_list.append(pred_batch)
        # get r2 per index
        ss_res, ss_tot = coeff_of_deter_per_index(target_batch, pred_batch)
        batch_ss_res[batch_idx, :] = ss_res
        batch_ss_tot[batch_idx, :] = ss_tot
        min_dev, max_dev, mean_dev = eval_abs_deviation(target_batch, pred_batch)
        batch_min_dev[batch_idx, :] = min_dev
        batch_max_dev[batch_idx, :] = max_dev
        batch_mean_dev[batch_idx, :] = mean_dev
        per_min_dev, per_max_dev, per_mean_dev = eval_per_abs_deviation(target_batch, pred_batch)
        batch_per_min_dev[batch_idx, :] = per_min_dev
        batch_per_max_dev[batch_idx, :] = per_max_dev
        batch_per_mean_dev[batch_idx, :] = per_mean_dev
        
    r2 = 1- ( np.sum(batch_ss_res, axis=0) / (np.sum(batch_ss_tot, axis=0) + K.epsilon()) )# np.mean(batch_r2, axis=0)
    
    min_dev = np.min(batch_min_dev, axis=0)
    max_dev = np.max(batch_max_dev, axis=0)
    mean_dev = np.mean(batch_mean_dev, axis=0)
    
    per_min_dev = np.min(batch_per_min_dev, axis=0)
    per_max_dev = np.max(batch_per_max_dev, axis=0)
    per_mean_dev = np.mean(batch_per_mean_dev, axis=0)
    
    target_zer = np.concatenate(target_zer_list, axis=0)
    pred_zer = np.concatenate(pred_zer_list, axis=0)
    print('shapes of target & pred coeff:', target_zer.shape, pred_zer.shape)
    zernike_coef = {'target':target_zer, 'predicted': pred_zer}
    
    metrics = {'r2': r2, 
               'min_dev': min_dev, 
               'max_dev': max_dev,
               'mean_dev': mean_dev,
               'per_min_dev': per_min_dev,
               'per_max_dev': per_max_dev,
               'per_mean_dev': per_mean_dev,
               'zernike': zernike_coef}
    
    return metrics
        
def eval_reg_model(config_fname, model_fname, split_set, gpu_ids=0, gpu_mem_frac=0.95):
    """Eval model performance"""

    with open(config_fname, 'r') as f:
        config = yaml.load(f)
    
    split_fname = os.path.join(config['trainer']['model_dir'], 'split_idx.pkl')
    with open(split_fname, 'rb') as f:
        split_fnames = pickle.load(f)
    sess = train_utils.set_keras_session(gpu_ids=gpu_ids,
                                         gpu_mem_frac=gpu_mem_frac)
    model = load_model(config, model_fname)
    metrics = predict_coef(model, split_fnames[split_set], 
                           config['network']['num_focal_planes'],
                           config['trainer']['batch_size'],
                           config['network']['regression_length'])
    np.savez(os.path.join(config['trainer']['model_dir'], 'zernike.npz'),
             target=metrics['zernike']['target'], 
             predicted=metrics['zernike']['predicted'])
    
    print('val:', np.around(metrics['r2'], 6))
    print('min:', np.around(metrics['min_dev'], 6))
    print('max:', np.around(metrics['max_dev'], 6))
    print('mean:', np.around(metrics['mean_dev'], 6))
    print('per_min:', np.around(metrics['per_min_dev'], 6))
    print('per_max:', np.around(metrics['per_max_dev'], 6))
    print('per_mean:', np.around(metrics['per_mean_dev'], 6))
    
    
    
