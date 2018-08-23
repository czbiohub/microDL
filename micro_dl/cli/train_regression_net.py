#!/usr/bin/env python
import argparse
import glob
import natsort
import numpy as np
import os
import pickle
import yaml

from micro_dl.train.trainer import BaseKerasTrainer
from micro_dl.utils.train_utils import check_gpu_availability, \
    split_train_val_test
from micro_dl.input.dataset_regression import RegressionDataSet


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help=('specify the gpu to use: 0,1,...',
                              ', -1 for debugging'))
    parser.add_argument('--gpu_mem_frac', type=float, default=1.,
                        help='specify the gpu memory fraction to use')
    parser.add_argument('--config', type=str,
                       help='path to yaml configuration file')
    args = parser.parse_args()
    return args


def train(config_fname, gpu_id, gpu_mem_frac):
    """Run training"""

    #  read config
    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    try:
        input_dir = os.path.join(config['dataset']['data_dir'], 'TrainingData')
        fnames = natsort.natsorted(glob.glob(os.path.join(input_dir, '*.npy')))
        #  save fnames as a csv to read train, val, test split later
        assert len(fnames) > 0, 'input_dir does not contain any files'
    except IOError as e:
        e.args += 'cannot read images in input_dir'
        raise
    num_samples = len(fnames)

    #  train, test, val split. val is a must and not optional
    train_ratio = config['dataset']['split_ratio']['train']
    test_ratio = config['dataset']['split_ratio']['test']
    val_ratio = config['dataset']['split_ratio']['val']

    split_indices = split_train_val_test(num_samples, train_ratio,
                                         test_ratio, val_ratio)

    fnames = np.array(fnames)
    train_fnames = fnames[split_indices['train']]
    val_fnames = fnames[split_indices['val']]
    test_fnames = fnames[split_indices['test']]
    split_fnames= {}
    split_fnames['train'] = train_fnames
    split_fnames['val'] = val_fnames
    split_fnames['test'] = test_fnames

    #  save fnames for later use, esp for evaluating performance on test
    split_fname = os.path.join(config['trainer']['model_dir'],
                                      'split_idx.pkl')

    with open(split_fname, 'wb') as f:
        pickle.dump(split_indices, f)

    # read the regression coeff
    regress_values = np.load(os.path.join(config['dataset']['data_dir'],
                                          'zernikecoeff.npy'))

    #  create dataset/generator for train/test set
    num_focal_planes = config['network']['num_focal_planes']
    batch_size = config['trainer']['batch_size']
    train_dataset = RegressionDataSet(
        input_fnames=train_fnames, num_focal_planes=num_focal_planes,
        regression_coeff=regress_values[split_indices['train']],
        batch_size=batch_size
    )
    val_dataset = RegressionDataSet(
        input_fnames=val_fnames, num_focal_planes=num_focal_planes,
        regression_coeff=regress_values[split_indices['val']],
        batch_size=batch_size)

    #  start training
    trainer = BaseKerasTrainer(config=config,
                               model_dir=config['trainer']['model_dir'],
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               gpu_ids=gpu_id,
                               gpu_mem_frac=gpu_mem_frac)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    gpu_available = False
    if args.gpu >= 0:
        gpu_available = check_gpu_availability(args.gpu, args.gpu_mem_frac)
    if not isinstance(args.gpu, int):
        raise NotImplementedError
    if gpu_available:
        train(args.config, args.gpu, args.gpu_mem_frac)
