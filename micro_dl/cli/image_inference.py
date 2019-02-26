#!/usr/bin/env/python
"""Model inference on large images"""
import argparse
import cv2
import natsort
import numpy as np
import os
import pandas as pd
import time
import yaml
import keras.backend as K

import micro_dl.plotting.plot_utils as plot_utils
import micro_dl.train.model_inference as inference
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
from micro_dl.utils.tile_utils import preprocess_imstack
from micro_dl.utils.train_utils import select_gpu, get_metrics, get_loss
import micro_dl.utils.train_utils as train_utils


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=None,
                        help=('Optional: specify the gpu to use: 0,1,...',
                              ', -1 for debugging. Default: pick best GPU'))
    parser.add_argument('--gpu_mem_frac', type=float, default=None,
                        help='Optional: specify the gpu memory fraction to use')

    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Directory containing model weights, config and csv files',
    )
    parser.add_argument(
        '--model_fname',
        type=str,
        default=None,
        help='File name of weights in model dir (.hdf5). If None grab newest.',
    )
    parser.add_argument(
        '--test_data',
        dest='test_data',
        action='store_true',
        help="Use test indices in split_samples.json",
    )
    parser.add_argument(
        '--all_data',
        dest='test_data',
        action='store_false',
    )
    parser.set_defaults(test_data=True)
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing images',
    )
    parser.add_argument(
        '--ext',
        type=str,
        default='.png',
        help='Image extension, if .npy, image is saved as is, else scaled to uint16'
    )
    parser.add_argument(
        '--save_figs',
        type=bool,
        default=False,
        help='Saves input, target, prediction plots. Assumes you have target channel'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default=['coeff_determination', 'ssim'],
        nargs='*',
        help='Metrics for predicted images'
    )
    args = parser.parse_args()
    return args


def run_prediction(args, gpu_ids, gpu_mem_frac):
    """
    Predict images given model + weights.
    If the test_data flag is set to True, the test indices in
    split_samples.json file in model directory will be predicted
    Otherwise, all images in image directory will be predicted.
    It will load the config.yml file save in model_dir to reconstruct the model.
    Predictions are converted to uint16 and saved as png as default, but can
    also be saved as is in .npy format.
    If saving figures, it assumes that input as well as target channels are
    present in image_dir.
    """
    if gpu_ids >= 0:
        sess = train_utils.set_keras_session(
            gpu_ids=gpu_ids, gpu_mem_frac=gpu_mem_frac
        )
    # Load config file
    config_name = os.path.join(args.model_dir, 'config.yml')
    with open(config_name, 'r') as f:
        config = yaml.load(f)
    # Load frames metadata and determine indices
    network_config = config['network']
    dataset_config = config['dataset']
    trainer_config = config['trainer']
    frames_meta = pd.read_csv(os.path.join(args.image_dir, 'frames_meta.csv'),
                              index_col=0)
    test_tile_meta = pd.read_csv(os.path.join(args.model_dir, 'test_metadata.csv'),
                              index_col=0)
    # TODO: generate test_frames_meta.csv together with tile csv during training
    test_frames_meta_filename = os.path.join(args.model_dir, 'test_frames_meta.csv')
    
    metrics = args.metrics
    if isinstance(metrics, str):
        metrics = [metircs]
    loss = trainer_config['loss']
    metrics_cls = get_metrics(args.metrics)
    loss_cls = get_loss(loss)
    split_idx_name = dataset_config['split_by_column']
    K.set_image_data_format(network_config['data_format'])
    if args.test_data:
        idx_fname = os.path.join(args.model_dir, 'split_samples.json')
        try:
            split_samples = aux_utils.read_json(idx_fname)
            test_ids = split_samples['test']
        except FileNotFoundError as e:
            print("No split_samples file. Will predict all images in dir.")
    else:
        test_ids = np.unique(frames_meta[split_idx_name])

    # Find other indices to iterate over than split index name
    # E.g. if split is position, we also need to iterate over time and slice
    metadata_ids = {split_idx_name: test_ids}
    iter_ids = ['slice_idx', 'pos_idx', 'time_idx']
    n_test_row = len(test_ids)
    for id in iter_ids:
        if id != split_idx_name:
            metadata_ids[id] = np.unique(test_tile_meta[id])
            n_test_row *= len(metadata_ids[id])
    test_frames_meta = pd.DataFrame(columns=
                                    frames_meta.columns.values.tolist()+metrics,
                                    index=range(n_test_row))
    # Get model weight file name, if none, load latest saved weights
    model_fname = args.model_fname
    if model_fname is None:
        fnames = [f for f in os.listdir(args.model_dir) if f.endswith('.hdf5')]
        assert len(fnames) > 0, 'No weight files found in model dir'
        fnames = natsort.natsorted(fnames)
        model_fname = fnames[-1]
    weights_path = os.path.join(args.model_dir, model_fname)

    # Create image subdirectory to write predicted images
    pred_dir = os.path.join(args.model_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    # If saving figures, create another subdirectory to predictions
    if args.save_figs:
        fig_dir = os.path.join(pred_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

    # If network depth is > 3 determine depth margins for +-z
    depth = 1
    if 'depth' in config['network']:
        depth = config['network']['depth']
        if depth > 1:
            metadata_ids['slice_idx'] = aux_utils.adjust_slice_margins(
                slice_ids=metadata_ids['slice_idx'],
                depth=depth,
            )
    # Get input channel
    # TODO: Add multi channel support once such models are tested
    input_channel = config['dataset']['input_channels'][0]
    target_channel = config['dataset']['target_channels'][0]
    assert isinstance(input_channel, int),\
        "Only supporting single input channel for now"
    # Get data format
    data_format = 'channels_first'
    if 'data_format' in config['network']:
        data_format = config['network']['data_format']
    # Load model with predict = True
    model = inference.load_model(
        network_config=config['network'],
        model_fname=weights_path,
        predict=True,
    )
    print(model.summary())
    model.compile(loss=loss_cls, optimizer='Adam', metrics=metrics_cls)
    test_row_ind = 0
    # Iterate over all indices for test data
    for time_idx in metadata_ids['time_idx']:
        for pos_idx in metadata_ids['pos_idx']:
            for slice_idx in metadata_ids['slice_idx']:
                # TODO: Add flatfield support
                im_stack = preprocess_imstack(
                    frames_metadata=frames_meta,
                    input_dir=args.image_dir,
                    depth=depth,
                    time_idx=time_idx,
                    channel_idx=input_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                )
                # Crop image shape to nearest factor of two
                im_stack = image_utils.crop2base(im_stack)
                # Remove singular z dimension for 2D image
                im_stack = np.squeeze(im_stack)
                # Change 3D image dimension order to zyx (3D)
                if depth > 1:
                    im_stack = np.transpose(im_stack, [2, 0, 1])
                # Add channel dimension
                if data_format == 'channels_first':
                    im_stack = im_stack[np.newaxis,...]
                else:
                    im_stack = im_stack[..., np.newaxis]

                # add batch dimensions
                im_stack = np.expand_dims(im_stack, axis=0)
                # Predict on large image
                start = time.time()
                im_pred = inference.predict_on_larger_image(
                    model=model,
                    input_image=im_stack,
                )
                print("Inference time:", time.time() - start)
                # Write prediction image
                im_name = aux_utils.get_im_name(
                    time_idx=time_idx,
                    channel_idx=input_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                    ext=args.ext,
                )
                file_name = os.path.join(pred_dir, im_name)
                if args.ext == '.png':
                    # Convert to uint16 for now
                    im_pred = 2 ** 16 * (im_pred - im_pred.min()) / \
                              (im_pred.max() - im_pred.min())
                    im_pred = im_pred.astype(np.uint16)
                    cv2.imwrite(file_name, np.squeeze(im_pred))
                if args.ext == '.tif':
                    # Convert to float32 and remove batch dimension
                    im_pred = im_pred.astype(np.float32)
                    cv2.imwrite(file_name, np.squeeze(im_pred))
                elif args.ext == '.npy':
                    np.save(file_name, im_pred, allow_pickle=True)
                else:
                    raise ValueError('Unsupported file extension')
                # Save figures if specified

                # Load target
                meta_idx = aux_utils.get_meta_idx(
                    frames_meta,
                    time_idx,
                    target_channel,
                    slice_idx,
                    pos_idx,
                )
                file_path = os.path.join(
                    args.image_dir,
                    frames_meta.loc[meta_idx, "file_name"],
                )
                test_frames_meta_row = frames_meta.loc[[meta_idx]]
                im_target = preprocess_imstack(
                    frames_metadata=frames_meta,
                    input_dir=args.image_dir,
                    depth=1,
                    time_idx=time_idx,
                    channel_idx=target_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                )
                im_target = image_utils.crop2base(im_target)
                if data_format == 'channels_first':
                    # Change dimension order to zyx (3D) or channel first (2D)
                    im_target = np.transpose(im_target, [2, 0, 1])
                im_target = im_target[np.newaxis, ...]

                metric_vals = model.evaluate(x=im_stack, y=im_target)
                for metric, metric_val in zip([loss]+metrics, metric_vals):
                    test_frames_meta_row[metric] = metric_val
                print(test_frames_meta_row)
                test_frames_meta.loc[test_row_ind] = test_frames_meta_row.loc[meta_idx]
                test_row_ind += 1
                if args.save_figs:
                    # assuming target and predicted images are always 2D for now
                    plot_utils.save_predicted_images(
                        input_batch=im_stack,
                        target_batch=im_target,
                        pred_batch=im_pred,
                        output_dir=fig_dir,
                        output_fname=im_name[:-4],
                        ext='jpg',
                        tol=1,
                        font_size=15
                    )
    # test_frames_meta = pd.concat(test_frames_meta, ignore_index=True)
    test_frames_meta = test_frames_meta.append(
        test_frames_meta[metrics].agg('mean'), ignore_index=True)
    test_frames_meta.to_csv(test_frames_meta_filename, sep=",")


if __name__ == '__main__':
    args = parse_args()
    # Get GPU ID and memory fraction
    gpu_id, gpu_mem_frac = select_gpu(
        args.gpu,
        args.gpu_mem_frac,
    )
    run_prediction(args, gpu_id, gpu_mem_frac)
