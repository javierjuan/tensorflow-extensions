import keras

from .feedforward import FeedForward
from .pooling import ChannelMaxPooling, ChannelAveragePooling


@keras.saving.register_keras_serializable(package='tfe.layers')
class ConvolutionalAttention2D(keras.layers.Layer):
    def __init__(self,
                 layer,
                 reduction_factor=8,
                 kernel_size=(7, 7),
                 activation='mish',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 data_format=None,
                 convolution_groups=1,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduction_factor = reduction_factor
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kernel_size = kernel_size
        self.data_format = data_format
        self.convolution_groups = convolution_groups
        self.supports_masking = True

        self._layer = layer
        self._channel_attention = ChannelAttention2D(
            reduction_factor=reduction_factor, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self._spatial_attention = SpatialAttention2D(
            kernel_size=kernel_size, data_format=data_format, convolution_groups=convolution_groups, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)

    def call(self, inputs, training=False, **kwargs):
        x = self._layer(inputs, training=training)
        x = self._channel_attention(x, training=training)
        x = self._spatial_attention(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = self._layer.compute_output_shape(input_shape=input_shape)
        output_shape = self._channel_attention.compute_output_shape(input_shape=output_shape)
        return self._spatial_attention.compute_output_shape(input_shape=output_shape)

    @classmethod
    def from_config(cls, config):
        config['layer'] = keras.layers.deserialize(config['layer'])
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer': keras.layers.serialize(self._layer),
            'reduction_factor': self.reduction_factor,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'data_format': self.data_format,
            'convolution_groups': self.convolution_groups
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class ChannelAttention2D(keras.layers.Layer):
    def __init__(self,
                 reduction_factor=8,
                 activation='mish',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 axis=-1,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduction_factor = reduction_factor
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.axis = axis
        self.supports_masking = True

        self._feed_forward = None
        self._reshape = None
        self._global_max_pooling = keras.layers.GlobalMaxPooling2D()
        self._global_average_pooling = keras.layers.GlobalAveragePooling2D()
        self._concatenate = keras.layers.Concatenate(axis=axis)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self._feed_forward = FeedForward(
            units=[input_channels * 2 // self.reduction_factor, input_channels],
            activation=[self.activation, 'sigmoid'], use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint)
        self._reshape = keras.layers.Reshape(target_shape=(1, 1, input_channels))
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self._global_average_pooling(inputs)
        y = self._global_max_pooling(inputs)
        z = self._concatenate([x, y])
        attention = self._feed_forward(z)
        attention = self._reshape(attention)
        return inputs * attention

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_factor': self.reduction_factor,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'axis': self.axis
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class SpatialAttention2D(keras.layers.Layer):
    def __init__(self,
                 kernel_size=(7, 7),
                 data_format=None,
                 convolution_groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name=None,
                 axis=-1,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.data_format = data_format
        self.convolution_groups = convolution_groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.axis = axis
        self.supports_masking = True

        self._convolution = keras.layers.Convolution2D(
            filters=1, kernel_size=kernel_size, strides=(1, 1), padding='same', data_format=data_format,
            dilation_rate=(1, 1), groups=convolution_groups, activation='sigmoid', use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self._channel_max_pooling = ChannelMaxPooling(keepdims=True)
        self._channel_average_pooling = ChannelAveragePooling(keepdims=True)
        self._concatenate = keras.layers.Concatenate(axis=axis)

    def call(self, inputs, **kwargs):
        cap = self._channel_average_pooling(inputs)
        cmp = self._channel_max_pooling(inputs)
        attention = self._concatenate([cap, cmp])
        attention = self._convolution(attention)
        return inputs * attention

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'data_format': self.data_format,
            'convolution_groups': self.convolution_groups,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'axis': self.axis
        })
        return config
