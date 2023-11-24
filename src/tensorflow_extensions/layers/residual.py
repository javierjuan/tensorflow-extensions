import keras_core as keras

from .attention import ConvolutionalAttention2D


@keras.saving.register_keras_serializable(package='tfe.layers')
class ResidualBlock2D(keras.layers.Layer):
    def __init__(self,
                 layer,
                 attention=True,
                 reduction_factor=8,
                 kernel_size=(7, 7),
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
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
        self.layer = layer if isinstance(layer, keras.layers.Layer) else keras.layers.deserialize(layer)
        self.attention = attention
        self.reduction_factor = reduction_factor
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
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

        self.attention_block = ConvolutionalAttention2D(
            reduction_factor=reduction_factor, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            synchronized=synchronized, axis=axis, kernel_size=kernel_size, strides=strides, padding=padding,
            data_format=data_format, dilation_rate=dilation_rate, convolution_groups=convolution_groups) if attention \
            else None
        self.add = keras.layers.Add()
        self.adapt_input = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        layer_channels = self.layer.compute_output_shape(input_shape=input_shape)[-1]
        if input_channels != layer_channels:
            self.adapt_input = keras.layers.Convolution2D(
                filters=layer_channels, kernel_size=(1, 1), padding='same', activation=None)
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self.layer(inputs, training=training)
        if self.attention_block is not None:
            x = self.attention_block(x, training=training)
        x = self.add([x, self.adapt_input(inputs, training=training) if self.adapt_input is not None else inputs])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer': keras.layers.serialize(self.layer),
            'attention': self.attention,
            'reduction_factor': self.reduction_factor,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'convolution_groups': self.convolution_groups,
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
