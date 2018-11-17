"""Custom losses"""
from keras.objectives import *
from keras import backend as K
from keras.losses import mean_squared_error, mean_absolute_error
import micro_dl.keras_contrib.backend as KC
import tensorflow as tf
import micro_dl.train.metrics as metrics


def loss_DSSIM(y_true, y_pred):
    """Need tf0.11rc to work"""
    y_true = tf.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = tf.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))
    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    patches_true = tf.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    u_true = K.mean(patches_true, axis=3)
    u_pred = K.mean(patches_pred, axis=3)
    var_true = K.var(patches_true, axis=3)
    var_pred = K.var(patches_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    return K.mean(((1.0 - ssim) / 2))

class DSSIM_Loss():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Structural dissimilarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        :param k1: Parameter of the SSIM (default 0.01)
        :param k2: Parameter of the SSIM (default 0.03)
        :param kernel_size: Size of the sliding window (default 3)
        :param max_value: Max value of the output (default 1.0)
        source: https://github.com/keras-team/keras_contrib/blob/master/keras_contrib/losses/dssim.py
        """

        self.__name__ = 'DSSIM_Loss'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = KC.image_data_format()
        self.backend = KC.backend()

    def __int_shape(self, x):
        return KC.int_shape(x) if self.backend == 'tensorflow' else KC.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [self.kernel_size, self.kernel_size]
        y_true = KC.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = KC.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, 'valid', self.dim_ordering)
        patches_true = KC.extract_image_patches(y_true, kernel, kernel, 'valid', self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = KC.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = KC.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = KC.mean(patches_true, axis=-1)
        u_pred = KC.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)


def dssim_loss(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    ssim = tf.image.ssim(y_true, y_pred, max_val=10)
    return K.mean((1.0 - ssim) / 2.0) + 0.2 * K.mean(mae)

def mean_sqrt_error(y_true, y_pred):
    loss = K.sqrt(mean_absolute_error(y_true, y_pred))
    return K.mean(loss)

def mse_binary_wtd(n_channels):
    """Converts a loss function into weighted loss function

    https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
    https://github.com/keras-team/keras/issues/3270
    https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred

    nested functions -> closures
    A Closure is a function object that remembers values in enclosing
    scopes even if they are not present in memory. Read only access!!

    :mask_image: a binary image (assumes foreground / background classes)
    :return: weighted loss
    """

    def mse_wtd(y_true, y_pred):
        try:
            y_true, mask_image = tf.split(y_true, [n_channels, 1], axis=1)
        except Exception as e:
            print('cannot separate mask and y_true' + str(e))
  
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        weights = K.batch_flatten(mask_image)
        weights = K.cast(weights, 'float32')
        loss = K.square(y_pred - y_true)

        fg_count = K.sum(weights, axis=1)
        total_count = K.cast(K.shape(y_true)[1], 'float32')
        fg_volume_fraction = tf.div(fg_count, total_count)
        bg_volume_fraction = 1-fg_volume_fraction
        # fg_vf is a tensor
        fg_weights = tf.where(fg_volume_fraction >= 0.5,
                              fg_volume_fraction, bg_volume_fraction)
        fg_mask = weights * K.expand_dims(fg_weights, axis=1)
        bg_mask = (1 - weights) * K.expand_dims(1 - fg_weights, axis=1)
        mask = fg_mask + bg_mask
        modified_loss = K.mean(K.sum(loss * mask, axis=1))
        return modified_loss
    return mse_wtd


def dice_coef_loss(y_true, y_pred):
    """
    The Dice loss function is defined by 1 - DSC
    since the DSC is in the range [0,1] where 1 is perfect overlap
    and we're looking to minimize the loss.

    :param y_true: true values
    :param y_pred: predicted values
    :return: Dice loss
    """
    return 1. - metrics.dice_coef(y_true, y_pred)



