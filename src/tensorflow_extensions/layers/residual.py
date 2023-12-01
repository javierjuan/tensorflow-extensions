import keras_core as keras

from .attention import ConvolutionalAttention2D


@keras.saving.register_keras_serializable(package='tfe.layers')
class ResidualBlock2D(keras.layers.Layer):
    def __init__(self,
                 layer,
                 attention=True,
                 reduction_factor=8,
                 kernel_size=(7, 7),
                 data_format=None,
                 convolution_groups=1,
                 activation='mish',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer = layer if isinstance(layer, keras.layers.Layer) else keras.layers.deserialize(layer)
        self.attention = attention
        self.reduction_factor = reduction_factor
        self.kernel_size = kernel_size
        self.data_format = data_format
        self.convolution_groups = convolution_groups
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.supports_masking = True

        self.attention_block = ConvolutionalAttention2D(
            reduction_factor=reduction_factor, kernel_size=kernel_size, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, data_format=self.layer.data_format,
            convolution_groups=self.layer.convolution_groups) if attention else None
        self.add = keras.layers.Add()
        self.convolution = None

    def build(self, input_shape):
        output_shape = self.layer.compute_output_shape(input_shape=input_shape)
        if input_shape[-1] != output_shape[-1]:
            self.convolution = keras.layers.Convolution2D(
                filters=output_shape[-1], kernel_size=(1, 1), activation=None, strides=self.layer.strides,
                padding='same', data_format=self.data_format, dilation_rate=(1, 1))
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self.layer(inputs, training=training)
        if self.attention_block is not None:
            x = self.attention_block(x, training=training)
        x = self.add([x, self.convolution(inputs, training=training) if self.convolution is not None else inputs])
        return x

    def compute_output_shape(self, input_shape):
        output_shape = self.layer.compute_output_shape(input_shape=input_shape)
        return self.attention_block.compute_output_shape(input_shape=output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer': keras.layers.serialize(self.layer),
            'attention': self.attention,
            'reduction_factor': self.reduction_factor,
            'kernel_size': self.kernel_size,
            'data_format': self.data_format,
            'convolution_groups': self.convolution_groups,
            'activation': self.activation,
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
