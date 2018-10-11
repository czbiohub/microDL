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
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )

def eval_abs_deviation(y_true, y_pred):
    """Absolute deviation"""
    
    abs_dev = np.abs(y_true - y_pred)
    min_dev = np.min(abs_dev, axis=0)
    max_dev = np.max(abs_dev, axis=0)
    mean_dev = np.mean(abs_dev, axis=0)
    return min_dev, max_dev, mean_dev

def eval_per_abs_deviation(y_true, y_pred):
    """Absolute deviation"""
    
    abs_dev = np.abs(y_true - y_pred) / (np.abs(y_true) + K.epsilon())
    min_dev = np.min(abs_dev, axis=0)
    max_dev = np.max(abs_dev, axis=0)
    mean_dev = np.mean(abs_dev, axis=0)
    return min_dev, max_dev, mean_dev
    
    
def predict_coef(model, input_fnames, num_focal_planes, batch_size, num_reg_coeff):
    """Predict zernike coeff"""
    
    reg_ds = RegressionDataSet(input_fnames, num_focal_planes, batch_size)
    num_batches = reg_ds.__len__()
    batch_r2 = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_min_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_max_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    batch_mean_dev = np.zeros((num_batches, num_reg_coeff), dtype='float')
    
    for batch_idx in range(num_batches):
        input_batch, target_batch = reg_ds.__getitem__(batch_idx) 
        pred_batch = model.predict(input_batch)
        # get r2 per index
        r2 = coeff_of_deter_per_index(target_batch, pred_batch)
        batch_r2[batch_idx, :] = r2
        min_dev, max_dev, mean_dev = eval_per_abs_deviation(target_batch, pred_batch)
        batch_min_dev[batch_idx, :] = min_dev
        batch_max_dev[batch_idx, :] = max_dev
        batch_mean_dev[batch_idx, :] = mean_dev
    cur_r2 = np.mean(batch_r2, axis=0)
    cur_min_dev = np.min(batch_min_dev, axis=0)
    cur_max_dev = np.max(batch_min_dev, axis=0)
    cur_mean_dev = np.mean(batch_min_dev, axis=0)
    return cur_r2, cur_min_dev, cur_max_dev, cur_mean_dev
        
def eval_reg_model(config_fname, model_fname, gpu_ids=0, gpu_mem_frac=0.95):
    """Eval model performance"""

    with open(config_fname, 'r') as f:
        config = yaml.load(f)
    
    split_fname = os.path.join(config['trainer']['model_dir'], 'split_idx.pkl')
    with open(split_fname, 'rb') as f:
        split_fnames = pickle.load(f)
    sess = train_utils.set_keras_session(gpu_ids=gpu_ids,
                                         gpu_mem_frac=gpu_mem_frac)
    model = load_model(config, model_fname)
    r2, min_dev, max_dev, mean_dev = predict_coef(
        model, split_fnames['val'], 
        config['network']['num_focal_planes'],
        config['trainer']['batch_size'],
        config['network']['regression_length']
    )
    print('val:', r2)
    print('min:', min_dev)
    print('max:', max_dev)
    print('mean:', mean_dev)
    
    
    
