"""Network for regressing vector of scalars from a set of images"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Dropout, Flatten, Input, Lambda,
                          MaxPool2D)
from keras.layers.merge import Add, Concatenate


class RegressionNet2D:
    """Network for regressing scalar values from a set of images"""

    def __init__(self, network_config):
        """Init

        :param dict network_config:
        """

        # [height, width, num_initial_filters/num_filters_per_block,
        # pooling_type, num_input_channels, data_format, regression_length]
        self.config = network_config
        assert network_config['height'] == network_config['width'], \
            'The network expects a square image'

        num_conv_blocks = int(np.log2(network_config['height']) -
                              np.log2(2) + 1)
        self.num_conv_blocks = num_conv_blocks

        if 'num_initial_filters' in network_config:
            assert 'num_filters_per_block' not in network_config, \
                'Both num_initial_filters & num_filters_per_block provided'
            num_init_filters = network_config['num_initial_filters']
            num_filters_per_block = [int(num_init_filters * 2 ** block_idx)
                                     for block_idx in range(num_conv_blocks)]
        elif 'num_filters_per_block' in network_config:
            num_filters_per_block = network_config['num_filters_per_block']
            assert len(num_filters_per_block) == num_conv_blocks, \
                '{} conv blocks != len(num_filters_per_block)'.\
                    format(num_conv_blocks)
        else:
            raise ValueError('Both num_initial_filters and '
                             'num_filters_per_block not in network_config')
        self.num_filters_per_block = num_filters_per_block

        pool_type = network_config['pooling_type']
        assert pool_type in ['max', 'average'], 'only max and average allowed'
        if pool_type == 'max':
            self.Pooling = MaxPool2D
        elif pool_type == 'average':
            self.Pooling = AveragePooling2D

        self.data_format = network_config['data_format']
        if network_config['data_format'] == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1

        dropout_prob = network_config['dropout']
        assert 0.0 < dropout_prob < 0.7, 'invalid dropout probability'
        self.dropout_prob = dropout_prob

    @staticmethod
    def _pad_channels(input_layer, num_desired_channels,
                      final_layer, channel_axis):
        """Zero pad along channels before residual/skip merge"""

        input_zeros = K.zeros_like(final_layer)
        num_input_layers = int(input_layer.get_shape()[channel_axis])
        new_zero_channels = int((num_desired_channels - num_input_layers) / 2)
        if num_input_layers % 2 == 0:
            zero_pad_layers = input_zeros[:, :new_zero_channels, :, :]
            layer_padded = Concatenate(axis=channel_axis)(
                [zero_pad_layers, input_layer, zero_pad_layers]
            )
        else:
            zero_pad_layers = input_zeros[:, :new_zero_channels + 1, :, :]
            layer_padded = Concatenate(axis=channel_axis)(
                [zero_pad_layers, input_layer, zero_pad_layers[:, :-1, :, :]]
            )
        return layer_padded

    def _merge_residual(self, final_layer, input_layer):
        """Add residual connection from input to last layer

        Residual layers are always added (no concat supported currently)
        :param keras.layers final_layer: last layer
        :param keras.layers input_layer: input_layer
        :return: input_layer 1x1 / padded to match the shape of final_layer
         and added
        """

        num_final_layers = int(final_layer.get_shape()[self.channel_axis])
        num_input_layers = int(input_layer.get_shape()[self.channel_axis])
        if num_input_layers > num_final_layers:
            # use 1x 1 to get to the desired num of feature maps
            input_layer = Conv2D(filters=num_final_layers,
                                 kernel_size=(1, 1),
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 data_format=self.data_format)(input_layer)
        if num_input_layers < num_final_layers:
            # padding with zeros along channels
            input_layer = Lambda(
                self._pad_channels,
                arguments={'num_desired_channels': num_final_layers,
                           'final_layer': final_layer,
                           'channel_axis': self.channel_axis})(input_layer)
        layer = Add()([final_layer, input_layer])
        return layer

    def _conv_block(self, layer, num_convs_per_block, filter_size,
                    num_filters, init='he_normal',
                    activation_type='relu', batch_norm=True,
                    dropout_prob=0.0, residual=False):
        """Convolution and down sampling block on the contracting path

        :param int block_idx: depth level of U-net
        :param keras.layers layer: input layer for the block
        :param int num_convs_per_block: specifies the number of repeated
         convolutions in each block
        :param int filter_size: size of the filter in xy or in-plane
        :param int num_filters: number of filters to be used
        :param str init: method used for initializing weights
        :param str activation_type: type of activation to be used
        :param bool batch_norm: indicator for batch norm
        :param float dropout_prob: as named
        :return: convolved layer
        """

        input_layer = layer
        for _ in range(num_convs_per_block):
            layer = Conv2D(filters=num_filters,
                           kernel_size=filter_size,
                           padding='same',
                           kernel_initializer=init,
                           data_format=self.data_format)(layer)
            if batch_norm:
                layer = BatchNormalization(axis=self.channel_axis)(layer)
            layer = Activation(activation_type)(layer)
            if dropout_prob:
                layer = Dropout(dropout_prob)(layer)

        if residual:
            layer = self._merge_residual(layer, input_layer)
        return layer

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        shape = (
            self.config['num_input_channels'],
            self.config['height'],
            self.config['width']
        )
        return shape

    def build_net(self):
        """Assemble the network"""

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # --------------------- Convolution blocks -------------------
        for block_idx in range(self.num_conv_blocks):
            block_name = 'conv_block_{}'.format(block_idx+1)
            with tf.name_scope(block_name):
                layer = self._conv_block(
                    layer=input_layer,
                    num_convs_per_block=self.config['num_convs_per_block'],
                    filter_size=self.config['filter_size'],
                    num_filters=self.num_filters_per_block[block_idx],
                    init='he_normal', activation_type='relu',
                    batch_norm=True, dropout_prob=0.0, residual=True)

            with tf.name_scope('pool_{}'.format(block_idx+1)):
                layer = self.Pooling(pool_size=2,
                                     data_format=self.data_format)(layer)
            input_layer = layer

        # --------------------- Dense blocks -------------------------
        num_units = self.num_filters_per_block[-1]
        regression_length = self.config['regression_length']

        if num_units / 16 > regression_length:
            dense_units = np.array([num_units / 2, num_units / 4,
                                    num_units / 8, num_units / 16],
                                   dtype='int')
        elif num_units / 8 > regression_length:
            dense_units = np.array([num_units / 2, num_units / 4,
                                    num_units / 8], dtype='int')
        elif num_units / 4 > regression_length:
            dense_units = np.array([num_units / 2, num_units / 4],
                                   dtype='int')
        else:
            raise ValueError('num features extracted <= 4 * regression_length')

        prev_dense_layer = Flatten()(layer)
        for dense_idx in range(len(dense_units)):
            block_name = 'dense_{}'.format(dense_idx + 1)
            with tf.name_scope(block_name):
                layer = Dense(dense_units[dense_idx],
                              kernel_initializer='he_normal',
                              activation='relu')(prev_dense_layer)
            prev_dense_layer = layer
        # --------------------- output block -------------------------
        with tf.name_scope('output'):
            outputs = Dense(regression_length,
                            kernel_initializer='he_normal',
                            activation='linear')(prev_dense_layer)
        return inputs, outputs
