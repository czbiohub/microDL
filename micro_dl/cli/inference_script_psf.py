#!/usr/bin/env/python
"""Model inference"""
import argparse
import os
import pandas as pd
import pickle
import yaml

from micro_dl.input.dataset_psf import DataSetForPSF
from micro_dl.train.model_inference import ModelEvaluator
from micro_dl.plotting.plot_utils import save_predicted_images_stack
from micro_dl.utils.train_utils import check_gpu_availability

def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help=('specify the gpu to use: 0,1,...',
                              ', -1 for debugging'))
    parser.add_argument('--gpu_mem_frac', type=float, default=0.99,
                        help='specify the gpu memory fraction to use')
    parser.add_argument('--config', type=str,
                       help='path to yaml configuration file')
    parser.add_argument('--model_fname', type=str, default=None,
                       help='fname with full path to model weights .hdf5')

    parser.add_argument('--num_batches', type=int, default=2,
                        help='run prediction on tiles for num_batches')

    parser.add_argument('--flat_field_correct', type=bool, default=True,
                        help='boolean indicator to correct for flat field')
    parser.add_argument('--focal_plane_idx', type=int, default=0,
                        help='idx for focal plane')

    args = parser.parse_args()
    return args


def run_inference(args):
    """Evaluate model performance"""

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    model_dir, _ = os.path.split(args.model_fname)
    # read the indices for test set
    split_fname = os.path.join(model_dir, 'split_fnames.pkl')
    with open(split_fname, 'rb') as f:
        split_fnames = pickle.load(f)

    test_fnames = split_fnames['test']
    num_focal_planes = config['network']['num_focal_planes']
    batch_size = config['trainer']['batch_size']
    test_dataset = DataSetForPSF(test_fnames, num_focal_planes, batch_size)

    ev_inst = ModelEvaluator(config,
                             model_fname=args.model_fname,
                             gpu_ids=args.gpu,
                             gpu_mem_frac=args.gpu_mem_frac)
    test_perf_metrics = ev_inst.evaluate_model(test_dataset)

    ev_inst.predict_on_tiles(test_dataset,
                             plot_fn=save_predicted_images_stack,
                             nb_batches=args.num_batches)

    return test_perf_metrics


if __name__ == '__main__':
    args = parse_args()
    gpu_available = False
    assert isinstance(args.gpu, int)
    if args.gpu >= 0:
        gpu_available = check_gpu_availability(args.gpu, args.gpu_mem_frac)
    if gpu_available:
        model_perf = run_inference(args)
        print('model performance on test images:', model_perf)


