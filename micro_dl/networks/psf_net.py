"""Implementation of an U-net for psf"""
import tensorflow as tf
from keras.layers import (
    Activation, AveragePooling3D, BatchNormalization, Conv3D, Dropout, Input,
    Lambda, MaxPool3D, UpSampling3D
)
from keras.layers.merge import Concatenate


class HybridUNet:
    """Implements a 3D u-net with n_focal_planes as slices"""

    def __init__(self, config):
        """Init

        :param dict config: a dict that contains network related params under
         the network key
        """

        self.config = config
        num_down_blocks = len(config['network']['num_filters_per_block'])
        self.num_down_blocks = num_down_blocks

        pool_type = config['network']['pooling_type']
        assert pool_type in ['max', 'average'], 'only max and average allowed'
        if pool_type == 'max':
            self.Pooling = MaxPool3D

        elif pool_type == 'average':
            self.Pooling = AveragePooling3D
        # no pool/downsampling along z
        self.pool_size = [2, 2, 1]

        data_format = config['network']['data_format']
        msg = 'Invalid data_format: channels_first / channels_last'
        assert data_format in ['channels_first', 'channels_last'], msg
        self.data_format = data_format
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 4

        dropout_prob = config['network']['dropout']
        assert 0.0 < dropout_prob < 0.7, 'invalid dropout probability'
        self.dropout_prob = dropout_prob

        num_focal_planes = config['network']['num_focal_planes']
        assert num_focal_planes <= 5, 'max num_focal_planes=5'
        self.num_focal_planes = num_focal_planes

    def _get_filter_shape(self, size_xy, size_z):
        """Get filter shape from in-plane size and # of focal planes

        :param int size_xy: filter size in xy-plane
        :param int size_z: number of focal planes
        :return: filter size as a list
        """

        msg = 'only int values allowed'
        assert isinstance(size_xy, int) and isinstance(size_z, int), msg
        if self.data_format == 'channels_first':
            filter_size = [size_z, size_xy, size_xy]
        else:
            filter_size = [size_xy, size_xy, size_z]
        return filter_size

    def _conv_block(self, layer, num_convs_per_block, filter_size,
                    num_filters, init='he_normal',
                    activation_type='relu', batch_norm=True,
                    dropout_prob=0.0):
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

        filter_size = self._get_filter_shape(filter_size,
                                             self.num_focal_planes)
        for _ in range(num_convs_per_block):
            layer = Conv3D(filters=num_filters,
                           kernel_size=filter_size,
                           padding='same',
                           kernel_initializer=init,
                           data_format=self.data_format)(layer)
            if batch_norm:
                layer = BatchNormalization(axis=self.channel_axis)(layer)
            layer = Activation(activation_type)(layer)
            if dropout_prob:
                layer = Dropout(dropout_prob)(layer)
        return layer

    def _pool_block(self, layer):
        """Applies pooling to layers in the contracting path

        :param keras.layers layer: input layer for pooling / down-sampling
        """

        layer = self.Pooling(pool_size=self.pool_size,
                             data_format=self.data_format)(layer)
        return layer

    def _skip_block(self, layer):
        """Downsamples skip layers along Z

        The contracting path of this U-net uses 3D images of shape
        [x, y, num of focal planes]. The expanding path reduces the shape to
        [x, y, 1] as we use num of focal plane images to predict the center
        image

        :param keras.layers layer: layers to be used in skip connection
        :return: convolved layer with valid padding
        """

        filter_size = self._get_filter_shape(1, 1)
        num_filters = 1
        # switch z and channel axis
        if self.data_format == 'channels_first':
            layer = Lambda(lambda x: tf.transpose(x, [0, 4, 2, 3, 1]))(layer)
        else:
            layer = Lambda(lambda x: tf.transpose(x, [0, 1, 2, 4, 3]))(layer)
        layer = Conv3D(filters=num_filters, kernel_size=filter_size,
                       padding='valid', data_format=self.data_format)(layer)
        # switch back
        if self.data_format == 'channels_first':
            layer = Lambda(lambda x: tf.transpose(x, [0, 4, 2, 3, 1]))(layer)
        else:
            layer = Lambda(lambda x: tf.transpose(x, [0, 1, 2, 4, 3]))(layer)
        return layer

    def _upsampling_block(self, layer, skip_layers, num_convs_per_block,
                          filter_size, num_filters, init='he_normal',
                          batch_norm=True, activation_type='relu',
                          dropout_prob=0.0):
        """Upsampling, concatenate and convolve block in expanding path

        :param keras.layers layer: input layer to the block
        :param keras.layers skip_layers: layers to be used for skip connection
        """

        # no upsampling along z / focal planes
        up_filter_size = (2, 2, 1)
        upsampled_layer = UpSampling3D(size=up_filter_size,
                                       data_format=self.data_format)(layer)
        skip_layer_valid = self._skip_block(skip_layers)
        layer = Concatenate(axis=self.channel_axis)(
            [upsampled_layer, skip_layer_valid]
        )
        layer = self._conv_block(layer, num_convs_per_block, filter_size,
                                 num_filters, init=init,
                                 activation_type=activation_type,
                                 batch_norm=batch_norm,
                                 dropout_prob=dropout_prob)
        return layer

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        if self.data_format == 'channels_first':
            shape = (1, self.config['network']['width'],
                     self.config['network']['height'],
                     self.num_focal_planes)
        else:
            shape = (self.config['network']['width'],
                     self.config['network']['height'],
                     self.num_focal_planes, 1)
        return shape

    def build_net(self):
        """Assemble the network"""

        num_convs_per_block = self.config['network']['num_convs_per_block']
        filter_size = self.config['network']['filter_size']
        activation = self.config['network']['activation']
        batch_norm = self.config['network']['batch_norm']
        dropout_prob = self.config['network']['dropout']
        num_filters_per_block = self.config['network']['num_filters_per_block']

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling blocks ---------
        skip_layers_list = []
        for block_idx in range(self.num_down_blocks):
            block_name = 'down_block_{}'.format(block_idx + 1)
            with tf.name_scope(block_name):
                if block_idx > 0:
                    input_layer = self._pool_block(input_layer)
                layer = self._conv_block(
                    layer=input_layer,
                    num_convs_per_block=num_convs_per_block,
                    filter_size=filter_size,
                    num_filters=num_filters_per_block[block_idx],
                    batch_norm=batch_norm,
                    activation_type=activation,
                    dropout_prob=dropout_prob
                )
            if block_idx != len(num_filters_per_block):
                skip_layers_list.append(layer)
            input_layer = layer

        #  ---------- skip block before upsampling ---------
        block_name = 'skip_block_{}'.format(len(num_filters_per_block))
        with tf.name_scope(block_name):
            layer = self._skip_block(input_layer)
        input_layer = layer

        #  ---------- upsampling blocks ---------
        for block_idx in reversed(range(self.num_down_blocks - 1)):
            cur_skip_layers = skip_layers_list[block_idx]
            block_name = 'up_block_{}'.format(block_idx)
            with tf.name_scope(block_name):
                layer = self._upsampling_block(
                    layer=input_layer, skip_layers=cur_skip_layers,
                    num_convs_per_block=num_convs_per_block,
                    filter_size=filter_size,
                    num_filters=num_filters_per_block[block_idx],
                    activation_type=activation,
                    dropout_prob=dropout_prob)
            input_layer = layer

        # ------------ output block ------------------------
        final_activation = self.config['network']['final_activation']
        with tf.name_scope('output'):
            layer = Conv3D(filters=1, kernel_size=(1, 1, 1),
                           padding='same', kernel_initializer='he_normal',
                           data_format=self.data_format)(input_layer)
            outputs = Activation(final_activation)(layer)
        return inputs, outputs
