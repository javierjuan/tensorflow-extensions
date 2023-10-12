import tensorflow as tf

from .dense import DenseBlock
from .pooling import ChannelMaxPooling, ChannelAveragePooling


class ConvolutionalAttention2D(tf.keras.layers.Layer):
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
                 normalization='batch',
                 momentum=0.99,
                 epsilon=0.001,
                 normalization_groups=32,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 synchronized=False,
                 axis=-1,
                 kernel_size=(7, 7),
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
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
        self.normalization = normalization
        self.momentum = momentum
        self.epsilon = epsilon
        self.normalization_groups = normalization_groups
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.synchronized = synchronized
        self.axis = axis
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.convolution_groups = convolution_groups
        self.supports_masking = True

        self.channel_attention_block = ChannelAttention2D(
            reduction_factor=reduction_factor, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            synchronized=synchronized, axis=axis)
        self.spatial_attention_block = SpatialAttention2D(
            kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, convolution_groups=convolution_groups, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)

    def call(self, inputs, training=False, **kwargs):
        x = self.channel_attention_block(inputs, training=training)
        x = self.spatial_attention_block(x, training=training)
        return x

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
            'normalization': self.normalization,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'normalization_groups': self.normalization_groups,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'moving_mean_initializer': self.moving_mean_initializer,
            'moving_variance_initializer': self.moving_variance_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint,
            'synchronized': self.synchronized,
            'axis': self.axis,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'convolution_groups': self.convolution_groups
        })
        return config


class ChannelAttention2D(tf.keras.layers.Layer):
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
                 normalization='batch',
                 momentum=0.99,
                 epsilon=0.001,
                 normalization_groups=32,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 synchronized=False,
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
        self.normalization = normalization
        self.momentum = momentum
        self.epsilon = epsilon
        self.normalization_groups = normalization_groups
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.synchronized = synchronized
        self.axis = axis
        self.supports_masking = True

        self.dense_mid = None
        self.dense_out = None
        self.reshape = None
        self.global_max_pooling = tf.keras.layers.GlobalMaxPooling2D()
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.concatenate = tf.keras.layers.Concatenate(axis=axis)

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        input_channels = input_shape[-1]
        self.dense_mid = DenseBlock(
            units=input_channels * 2 // self.reduction_factor, activation=self.activation, use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint, normalization=self.normalization, momentum=self.momentum,
            epsilon=self.epsilon, normalization_groups=self.normalization_groups, center=self.center,
            scale=self.scale, beta_initializer=self.beta_initializer, gamma_initializer=self.gamma_initializer,
            moving_mean_initializer=self.moving_mean_initializer, gamma_regularizer=self.gamma_regularizer,
            moving_variance_initializer=self.moving_variance_initializer, beta_regularizer=self.beta_regularizer,
            beta_constraint=self.beta_constraint, gamma_constraint=self.gamma_constraint,
            synchronized=self.synchronized, axis=self.axis, rate=None, seed=None)
        self.dense_out = tf.keras.layers.Dense(
            units=input_channels, activation='sigmoid', use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)
        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, input_channels))

    def call(self, inputs, training=False, **kwargs):
        x = self.global_average_pooling(inputs)
        y = self.global_max_pooling(inputs)
        z = self.concatenate([x, y])
        x = self.dense_mid(z, training=training)
        attention = self.dense_out(x)
        attention = self.reshape(attention)
        return inputs * attention

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
            'normalization': self.normalization,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'normalization_groups': self.normalization_groups,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'moving_mean_initializer': self.moving_mean_initializer,
            'moving_variance_initializer': self.moving_variance_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint,
            'synchronized': self.synchronized,
            'axis': self.axis
        })
        return config


class SpatialAttention2D(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size=(7, 7),
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
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
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.convolution_groups = convolution_groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.supports_masking = True

        self.convolution = tf.keras.layers.Convolution2D(
            filters=1, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, groups=convolution_groups, activation='sigmoid', use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.channel_max_pooling = ChannelMaxPooling(keepdims=True)
        self.channel_average_pooling = ChannelAveragePooling(keepdims=True)
        self.concatenate = tf.keras.layers.Concatenate(axis=axis)

    def call(self, inputs, **kwargs):
        cap = self.channel_average_pooling(inputs)
        cmp = self.channel_max_pooling(inputs)
        attention = self.concatenate([cap, cmp])
        attention = self.convolution(attention)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'convolution_groups': self.convolution_groups,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint
        })
        return config
