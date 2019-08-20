import cv2
import natsort
import numpy as np
import os
import pandas as pd

from micro_dl.input.inference_dataset import InferenceDataset
import micro_dl.inference.model_inference as inference
from micro_dl.inference.evaluation_metrics import MetricsEstimator
from micro_dl.inference.stitch_predictions import ImageStitcher
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.image_utils import center_crop_to_shape
from micro_dl.utils.train_utils import set_keras_session
import micro_dl.utils.tile_utils as tile_utils

class InputStitcher:
    def __init__(self, config,
                 model_fname,
                 image_dir,
                 data_split,
                 image_param_dict,
                 gpu_id,
                 gpu_mem_frac,
                 metrics_list=None,
                 metrics_orientations=None,
                 mask_param_dict=None,
                 vol_inf_dict=None):
        """Init

        :param dict config: config dict with params related to dataset,
         trainer and network
        :param str model_fname: fname of the hdf5 file with model weights
        :param str image_dir: dir containing input images AND NOT TILES!
        :param dict image_param_dict: dict with keys image_format,
         flat_field_dir, im_ext, crop_shape. im_ext: npy or png or tiff.
         FOR 3D IMAGES USE NPY AS PNG AND TIFF ARE CURRENTLY NOT SUPPORTED.
         crop_shape: center crop the image to a specified shape before tiling
         for inference
        :params int gpu_id: gpu to use
        :params float gpu_mem_frac: Memory fractions to use corresponding
         to gpu_ids
        :param list metrics_list: list of metrics to estimate. available
         metrics: [ssim, corr, r2, mse, mae}]
         TODO: add accuracy and dice coeff to metrics list
        :param list metrics_orientations: xy, xyz, xz or yz
        :param dict mask_param_dict: dict with keys mask_dir and mask_channel
        :param dict vol_inf_dict: dict with params for 3D inference with keys:
         num_slices, inf_shape, tile_shape, num_overlap, overlap_operation.
         num_slices - in case of 3D, the full volume will not fit in GPU
         memory, specify the number of slices to use and this will depend on
         the network depth, for ex 8 for a network of depth 4. inf_shape -
         inference on a center sub volume. tile_shape - shape of tile for
         tiling along xyz. num_overlap - int for tile_z, list for tile_xyz
        """

        self.config = config
        self.data_format = self.config['network']['data_format']
        if gpu_id >= 0:
            sess = set_keras_session(gpu_ids=gpu_id,
                                     gpu_mem_frac=gpu_mem_frac)
            self.sess = sess

        # create network instance and load weights
        model_inst = self._create_model(model_fname)
        self.model_inst = model_inst
        self.frames_meta = pd.read_csv(os.path.join(image_dir,
                                               'frames_meta.csv'))
        assert 'image_format' in image_param_dict, \
            'image_format not in image_param_dict'
        assert 'im_ext' in image_param_dict, 'im_ext not in image_param_dict'

        flat_field_dir = None
        if 'flat_field_dir' in image_param_dict:
            flat_field_dir = image_param_dict['flat_field_dir']
        dataset_inst = InferenceDataset(
            image_dir=image_dir,
            dataset_config=config['dataset'],
            network_config=config['network'],
            df_meta=self.frames_meta,
            image_format=image_param_dict['image_format'],
            flat_field_dir=flat_field_dir
        )
        self.dataset_inst = dataset_inst
        self.image_format = image_param_dict['image_format']
        self.image_ext = image_param_dict['im_ext']
        crop_shape = None
        if 'crop_shape' in image_param_dict:
            crop_shape = image_param_dict['crop_shape']
        self.crop_shape = crop_shape

        # Create image subdirectory to write predicted images
        model_dir = config['trainer']['model_dir']
        pred_dir = os.path.join(model_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        self.pred_dir = pred_dir

        # create an instance of MetricsEstimator
        self.df_iteration_meta = dataset_inst.get_df_iteration_meta()

        metrics_est_inst = None
        if mask_param_dict is not None:
            assert ('mask_channel' in mask_param_dict and
                    'mask_dir' in mask_param_dict), \
                'Both mask_channel and mask_dir are needed'
        self.mask_param_dict = mask_param_dict

        metrics_est_inst = None
        if metrics_list is not None:
            metrics_est_inst = MetricsEstimator(
                metrics_list=metrics_list,
                masked_metrics=True,
            )
        self.metrics_est_inst = metrics_est_inst

        num_overlap = 0
        stitch_inst = None
        tile_option = None
        z_dim = 2
        if vol_inf_dict is not None:
            tile_option, num_overlap, stitch_inst, z_dim = \
                self._assign_vol_inf_options(image_param_dict,
                                             vol_inf_dict)
        self.tile_option = tile_option
        self.num_overlap = num_overlap
        self.stitch_inst = stitch_inst
        self.z_dim = z_dim
        self.vol_inf_dict = vol_inf_dict

        self.nrows = config['network']['width']
        self.ncols = config['network']['height']
        pos_orientations = ['xy', 'xyz', 'xz', 'yz']
        if metrics_orientations is not None:
            assert set(metrics_orientations).issubset(pos_orientations), \
                'orientation not in [xy, xyz, xz, yz]'
            self.df_xy = pd.DataFrame()
            self.df_xyz = pd.DataFrame()
            self.df_xz = pd.DataFrame()
            self.df_yz = pd.DataFrame()
        self.metrics_orientations = metrics_orientations

    def get_tile_indices(self):
        n_row = self.stitch_config['n_row']
        n_col = self.stitch_config['n_col']
        overlap = self.stitch_config['overlap']
        grid_pattern = self.stitch_config['grid_pattern']
        step_size = round(tile_size * (1-overlap))

        for row in range(0, n_rows - tile_size[0] + step_size[0], step_size[0]):
            if row + tile_size[0] > n_rows:
                row = check_in_range(row, n_rows, tile_size[0])
            for col in range(0, n_cols - tile_size[1] + step_size[1], step_size[1]):
                if col + tile_size[1] > n_cols:
                    col = check_in_range(col, n_cols, tile_size[1])
                img_id = 'r{}-{}_c{}-{}'.format(row, row + tile_size[0],
                                                col, col + tile_size[1])

                cur_index = (row, row + tile_size[0], col, col + tile_size[1])
                cropped_img = input_image[row: row + tile_size[0],
                              col: col + tile_size[1], ...]



    def stitch_input(self):
        crop_indices = None
        df_iteration_meta = self.dataset_inst.get_df_iteration_meta()
        pos_idx = df_iteration_meta['pos_idx'].unique()
        for idx, cur_pos_idx in enumerate(pos_idx):
            print(cur_pos_idx, ',{}/{}'.format(idx, len(pos_idx)))
            df_iter_meta_row_idx = df_iteration_meta.index[
                df_iteration_meta['pos_idx'] == cur_pos_idx
                ].values
            assert len(df_iter_meta_row_idx) == 1, \
                'more than one matching row found for position ' \
                '{}'.format(cur_pos_idx)
            cur_input, cur_target = \
                self.dataset_inst.__getitem__(df_iter_meta_row_idx[0])
            if self.crop_shape is not None:
                cur_input = center_crop_to_shape(cur_input,
                                                 self.crop_shape)
                cur_target = center_crop_to_shape(cur_target,
                                                  self.crop_shape)
            # assign zdim if not Unet2D
            if self.image_format == 'zyx':
                z_dim = 2 if self.data_format == 'channels_first' else 1
            elif self.image_format == 'xyz':
                z_dim = 4 if self.data_format == 'channels_first' else 3

            # stitch xy is the most common case
            tile_option = 'tile_xyz'
            overlap_dict = {
                'overlap_shape': num_overlap,
                'overlap_operation': vol_inf_dict['overlap_operation']
            }
            stitch_inst = ImageStitcher(
                tile_option=tile_option,
                overlap_dict=overlap_dict,
                image_format=self.image_format,
                data_format=self.data_format
            )

            step_size = (np.array(self.vol_inf_dict['tile_shape']) -
                         np.array(self.num_overlap))
            if crop_indices is None:
                # TODO tile_image works for 2D/3D imgs, modify for multichannel
                _, crop_indices = tile_utils.tile_image(
                    input_image=np.squeeze(cur_input),
                    tile_size=self.vol_inf_dict['tile_shape'],
                    step_size=step_size,
                    return_index=True
                )
            pred_block_list = self._predict_sub_block_xyz(cur_input,
                                                          crop_indices)
            input_image = self.stitch_inst.stitch_predictions(
                np.squeeze(cur_input).shape,
                pred_block_list,
                crop_indices
            )
            pred_image = np.squeeze(pred_image)
            target_image = np.squeeze(cur_target)
            # save prediction
            cur_row = self.df_iteration_meta.iloc[df_iter_meta_row_idx[0]]
            self.save_pred_image(predicted_image=pred_image,
                                 time_idx=cur_row['time_idx'],
                                 target_channel_idx=cur_row['channel_idx'],
                                 pos_idx=cur_row['pos_idx'],
                                 slice_idx=cur_row['slice_idx'])