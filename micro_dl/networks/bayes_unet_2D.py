"""Bayesian Unet 2D"""
from micro_dl.networks.base_unet import BaseUNet
import tensorflow as tf
from keras.layers import Activation, Input, UpSampling2D, UpSampling3D

from micro_dl.networks.base_conv_net import BaseConvNet
from micro_dl.networks.conv_blocks import conv_block,  residual_conv_block, \
    residual_downsample_conv_block, skip_merge
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.network_utils import get_keras_layer

class BayesUNet2D(BaseUNet):
    """2D Bayesian UNet

    [batch_size, num_channels, y, x] or [batch_size, y, x, num_channels]
    """

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        if self.config['data_format'] == 'channels_first':
            shape = (self.config['num_input_channels'],
                     self.config['height'],
                     self.config['width'])
        else:
            shape = (self.config['height'],
                     self.config['width'],
                     self.config['num_input_channels'])
        return shape

    def build_net(self):
        """Assemble the network"""

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling + middle blocks ---------
        skip_layers_list = []
        for block_idx in range(self.num_down_blocks + 1):
            block_name = 'down_block_{}'.format(block_idx + 1)
            with tf.name_scope(block_name):
                layer, cur_skip_layers = self._downsampling_block(
                    input_layer=input_layer, block_idx=block_idx
                )
            skip_layers_list.append(cur_skip_layers)
            input_layer = layer
        del skip_layers_list[-1]

        # ------------- Upsampling / decoding blocks -------------
        for block_idx in reversed(range(self.num_down_blocks)):
            cur_skip_layers = skip_layers_list[block_idx]
            block_name = 'up_block_{}'.format(block_idx)
            with tf.name_scope(block_name):
                layer = self._upsampling_block(input_layers=input_layer,
                                               skip_layers=cur_skip_layers,
                                               block_idx=block_idx)
            input_layer = layer

        # ------------ output block ------------------------
        final_activation = self.config['final_activation']
        # output mean and standard deviation for each channel
        num_output_channels = self.config['num_target_channels']
        conv_object = get_keras_layer(type='conv',
                                      num_dims=self.config['num_dims'])
        with tf.name_scope('output'):
            layer = conv_object(
                filters=num_output_channels,
                kernel_size=(1,) * self.config['num_dims'],
                padding=self.config['padding'],
                kernel_initializer=self.config['init'],
                data_format=self.config['data_format'])(input_layer)
            outputs = Activation(final_activation)(layer)
        return inputs, outputs