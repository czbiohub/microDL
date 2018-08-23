"""Dataset class for psf-net"""
import keras
import numpy as np
from micro_dl.input.dataset_psf import DataSetForPSF


class RegressionDataSet(DataSetForPSF):
    """Dataset class for generating input image and regression vector pairs"""

    def __init__(self, input_fnames, num_focal_planes, regression_coeff,
                 batch_size, shuffle=True, random_seed=42):
        """Init

        :param np.array input_fnames: vector containing fnames with full path
         of the simulated data in .npy format
        :param int num_focal_planes: number of focal planes acquired to model
         psf. (n=3 or 5)
        :param np.array regression_coeff: 2D array with z coefficients in the
         matching order of input_fnames
        :param int batch_size: number of datasets in each batch
        """

        super().__init__(input_fnames=input_fnames,
                         num_focal_planes=num_focal_planes,
                         batch_size=batch_size, shuffle=shuffle,
                         random_seed=random_seed)
        self.regression_coeff = regression_coeff

    def __getitem__(self, index):
        """Get a batch of data

        :param int index: batch index
        """

        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        if end_idx >= self.num_samples:
            end_idx = self.num_samples

        input_batch = []
        target_batch = []
        for idx in range(start_idx, end_idx, 1):
            cur_fname = self.input_fnames[self.row_idx[idx]]
            cur_input = np.load(cur_fname)
            cur_input = np.moveaxis(cur_input, -1, 0)
            cur_target = self.regression_coeff[self.row_idx[idx], :]
            input_batch.append(cur_input)
            target_batch.append(cur_target)
        input_batch = np.stack(input_batch)
        target_batch = np.stack(target_batch)
        return input_batch, target_batch
