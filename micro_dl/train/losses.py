"""Custom losses"""
from keras import backend as K
from keras.losses import mean_absolute_error
import tensorflow as tf

import micro_dl.train.metrics as metrics
from micro_dl.utils.aux_utils import get_channel_axis

# K.set_epsilon(1e-07)

def mae_loss(y_true, y_pred, mean_loss=True):
    """Mean absolute error

    Keras losses by default calculate metrics along axis=-1, which works with
    image_format='channels_last'. The arrays do not seem to batch flattened,
    change axis if using 'channels_first
    """
    if not mean_loss:
        return K.abs(y_pred - y_true)

    channel_axis = get_channel_axis(K.image_data_format())
    return K.mean(K.abs(y_pred - y_true), axis=channel_axis)


def mse_loss(y_true, y_pred, mean_loss=True):
    """Mean squared loss

    :param y_true: Ground truth
    :param y_pred: Prediction
    :return float: Mean squared error loss
    """
    if not mean_loss:
        return K.square(y_pred - y_true)

    channel_axis = get_channel_axis(K.image_data_format())
    return K.mean(K.square(y_pred - y_true), axis=channel_axis)


def kl_divergence_loss(y_true, y_pred):
    """KL divergence loss
    D(y||y') = sum(p(y)*log(p(y)/p(y'))

    :param y_true: Ground truth
    :param y_pred: Prediction
    :return float: KL divergence loss
    """
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    channel_axis = get_channel_axis(K.image_data_format())
    return K.sum(y_true * K.log(y_true / y_pred), axis=channel_axis)


def dssim_loss(y_true, y_pred):
    """Structural dissimilarity loss + L1 loss
    DSSIM is defined as (1-SSIM)/2
    https://en.wikipedia.org/wiki/Structural_similarity

    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :return float: 0.8 * DSSIM + 0.2 * L1
    """
    mae = mean_absolute_error(y_true, y_pred)
    return 0.8 * (1.0 - metrics.ssim(y_true, y_pred) / 2.0) + 0.2 * mae


def ms_ssim_loss(y_true, y_pred):
    """
    Multiscale structural dissimilarity loss + L1 loss
    Uses the same combination weight as the original paper by Wang et al.:
    https://live.ece.utexas.edu/publications/2003/zw_asil2003_msssim.pdf
    Tensorflow doesn't have a 3D version so for stacks the MS-SSIM is the
    mean of individual slices.

    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :return float: ms-ssim loss
    """
    mae = mae_loss(y_true, y_pred)
    return 0.84 * (1.0 - metrics.ms_ssim(y_true, y_pred)) + 0.16 * mae

def masked_loss(loss_fn, n_channels):
    """Converts a loss function to mask weighted loss function

    Loss is multiplied by mask. Mask could be binary, discrete or float.
    Provides different weighting of loss according to the mask.
    https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
    https://github.com/keras-team/keras/issues/3270
    https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred

    nested functions -> closures
    A Closure is a function object that remembers values in enclosing
    scopes even if they are not present in memory. Read only access!!
    Histogram and logical operators are not differentiable, avoid them in loss
    modified_loss = tf.Print(modified_loss, [modified_loss],
                             message='modified_loss', summarize=16)
    :param Function loss_fn: a loss function that returns a loss image to be
     multiplied by mask
    :param int n_channels: number of channels in y_true. The mask is added as
     the last channel in y_true
    :return function masked_loss_fn
    """

    def masked_loss_fn(y_true, y_pred):
        y_true, mask_image = metrics.split_tensor_channels(y_true, n_channels)
        loss = loss_fn(y_true, y_pred, mean_loss=False)
        total_loss = 0.0
        for ch_idx in range(n_channels):
            cur_loss = loss[:, ch_idx]
            cur_loss = cur_loss * mask_image
            mean_loss = K.mean(cur_loss)
            total_loss += mean_loss
        modified_loss = total_loss / n_channels
        return modified_loss
    return masked_loss_fn


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

def bnn_loss(n_channels):
    """wrapper function for Bayesian loss to pass n_channels as input
    :param n_channels:
    :return:
    """
    def bnn_loss_fn(y_true, y_pred):
        return weighted_mae_loss(y_true, y_pred, n_channels)
    return bnn_loss_fn

def weighted_mae_loss(y_true, y_pred, n_channels):
    """Bayesian loss that includes data uncertainty term
    """

    y_pred_mean, y_pred_std = metrics.split_tensor_channels(y_pred, n_channels, n_channels/2)
    mae_weighted = K.mean(K.abs(y_pred_mean - y_true) / (K.abs(y_pred_std) + K.epsilon()))
    std_reg = K.mean(K.log(2*K.abs(y_pred_std)))
    return mae_weighted + std_reg

def bnn_mse_loss(n_channels):
    """wrapper function for Bayesian loss to pass n_channels as input
    :param n_channels:
    :return:
    """
    def bnn_loss_fn(y_true, y_pred):
        return weighted_mse_loss(y_true, y_pred, n_channels)
    return bnn_loss_fn

def weighted_mse_loss(y_true, y_pred, n_channels):
    """Bayesian loss that includes data uncertainty term
    """
    limit = [-15, 15]
    y_pred_mean, y_pred_log_var = metrics.split_tensor_channels(y_pred, n_channels, n_channels/2)
    mse_weighted = K.mean(K.square(y_pred_mean - y_true) /
                          (2 * K.exp(K.clip(y_pred_log_var, limit[0], limit[1]))))
    # mse_weighted = K.mean(K.square(y_pred_mean - y_true))
    var_reg = 0.5 * K.mean(K.clip(y_pred_log_var, limit[0], limit[1]))
    return mse_weighted + var_reg
    # return mse_weighted
