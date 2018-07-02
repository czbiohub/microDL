"""Dataset class for psf-net"""
import keras
import numpy as np


class DataSetForPSF(keras.utils.Sequence):
    """Dataset class for generating input and target pairs"""

    def __init__(self, input_fnames, num_focal_planes, batch_size,
                 shuffle=True, random_seed=42):
        """Init

        :param np.array input_fnames: vector containing fnames with full path
         of the simulated data in .npy format
        :param int num_focal_planes: number of focal planes acquired to model
         psf. (n=3 or 5)
        :param int batch_size: number of datasets in each batch
        """

        self.input_fnames = input_fnames
        self.num_focal_planes = num_focal_planes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.num_samples = len(input_fnames)
        self.on_epoch_end()

    def __len__(self):
        """Gets the number of batches per epoch"""

        n_batches = int(self.num_samples / self.batch_size)
        return n_batches

    def on_epoch_end(self):
        """Update indices and shuffle after each epoch"""

        self.row_idx = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.row_idx)

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
            cur_image = np.load(cur_fname)
            cur_input = cur_image[:, :, :self.num_focal_planes]
            cur_input = np.expand_dims(cur_input, axis=0)
            cur_target = np.expand_dims(cur_image[:, :, -1], axis=0)
            cur_target = np.expand_dims(cur_target, axis=3)
            input_batch.append(cur_input)
            target_batch.append(cur_target)
        input_batch = np.stack(input_batch)
        target_batch = np.stack(target_batch)
        return input_batch, target_batch

