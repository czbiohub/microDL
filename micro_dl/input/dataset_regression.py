"""Dataset class for psf-net"""
import numpy as np
from micro_dl.input.dataset_psf import DataSetForPSF


class RegressionDataSet(DataSetForPSF):
    """Dataset class for generating input image and regression vector pairs"""

    def __init__(self, input_fnames, num_focal_planes, # regression_coeff,
                 batch_size, shuffle=True, random_seed=42, normalize=True):
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
        self.normalize = normalize
        # self.regression_coeff = regression_coeff

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
            # modified to work with the latest dataset with 21 zer terms
            input_stack = cur_input['images']
            blurred_stack = input_stack[:-1]
            if self.normalize:
                blurred_stack = blurred_stack - np.min(blurred_stack)
                blurred_stack = blurred_stack / np.max(blurred_stack)
                input_stack[:-1] = blurred_stack
            # input_stack = np.stack([center_image, unblurred_image])
            # blurred_stack = np.moveaxis(blurred_stack, -1, 0)
            # cur_target = self.regression_coeff[self.row_idx[idx], :]
            cur_target = cur_input['zernike'][1][3:] 
            # cur min and max z in dataset is -3.25 and 3.75
            cur_target = cur_target + 3.25
            cur_target = cur_target / 7.0
            input_batch.append(input_stack)
            target_batch.append(cur_target)
        input_batch = np.stack(input_batch)
        target_batch = np.stack(target_batch)
        target_batch = target_batch.astype('float32')
        return input_batch, target_batch
