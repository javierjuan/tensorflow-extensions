import keras

from .pooling import ChannelMaxPooling, ChannelAveragePooling


@keras.saving.register_keras_serializable(package='tfe.layers')
class ConvolutionalAttention2D(keras.layers.Layer):
    def __init__(self,
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

        self.channel_attention_block = ChannelAttention2D(
            reduction_factor=reduction_factor, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.spatial_attention_block = SpatialAttention2D(
            kernel_size=kernel_size, data_format=data_format, convolution_groups=convolution_groups, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)

    def call(self, inputs, training=False, **kwargs):
        x = self.channel_attention_block(inputs, training=training)
        x = self.spatial_attention_block(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = self.channel_attention_block.compute_output_shape(input_shape=input_shape)
        return self.spatial_attention_block.compute_output_shape(input_shape=output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
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

        self.dense_mid = None
        self.dense_out = None
        self.reshape = None
        self.global_max_pooling = keras.layers.GlobalMaxPooling2D()
        self.global_average_pooling = keras.layers.GlobalAveragePooling2D()
        self.concatenate = keras.layers.Concatenate(axis=axis)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.dense_mid = keras.layers.Dense(
            units=input_channels * 2 // self.reduction_factor, activation=self.activation, use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)
        self.dense_out = keras.layers.Dense(
            units=input_channels, activation='sigmoid', use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)
        self.reshape = keras.layers.Reshape(target_shape=(1, 1, input_channels))
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self.global_average_pooling(inputs)
        y = self.global_max_pooling(inputs)
        z = self.concatenate([x, y])
        x = self.dense_mid(z, training=training)
        attention = self.dense_out(x)
        attention = self.reshape(attention)
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

        self.convolution = keras.layers.Convolution2D(
            filters=1, kernel_size=kernel_size, strides=(1, 1), padding='same', data_format=data_format,
            dilation_rate=(1, 1), groups=convolution_groups, activation='sigmoid', use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.channel_max_pooling = ChannelMaxPooling(keepdims=True)
        self.channel_average_pooling = ChannelAveragePooling(keepdims=True)
        self.concatenate = keras.layers.Concatenate(axis=axis)

    def call(self, inputs, **kwargs):
        cap = self.channel_average_pooling(inputs)
        cmp = self.channel_max_pooling(inputs)
        attention = self.concatenate([cap, cmp])
        attention = self.convolution(attention)
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
