import keras_core as keras

from .convolution import ConvolutionBlock2D


@keras.saving.register_keras_serializable(package='tfe.layers')
class InceptionBlock2D(keras.layers.Layer):
    def __init__(self,
                 filters,
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
                 axis=-1,
                 rate=None,
                 seed=None,
                 pool_size=(3, 3),
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
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
        self.axis = axis
        self.rate = rate
        self.seed = seed
        self.pool_size = pool_size
        self.supports_masking = True

        self.block_1x = ConvolutionBlock2D(
            filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, convolution_groups=convolution_groups, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=None, seed=None)
        self.block_3x = ConvolutionBlock2D(
            filters=filters, kernel_size=(3, 3), strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, convolution_groups=convolution_groups, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=None, seed=None)
        self.block_5x0 = ConvolutionBlock2D(
            filters=filters, kernel_size=(3, 3), strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, convolution_groups=convolution_groups, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=None, seed=None)
        self.block_5x1 = ConvolutionBlock2D(
            filters=filters, kernel_size=(3, 3), strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, convolution_groups=convolution_groups, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=None, seed=None)
        self.max_pooling = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
        self.block_max_pooling = ConvolutionBlock2D(
            filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, convolution_groups=convolution_groups, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=None, seed=None)
        self.block_compression = ConvolutionBlock2D(
            filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, convolution_groups=convolution_groups, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=None, seed=None)
        self.concatenate = keras.layers.Concatenate(axis=axis)
        self.dropout = keras.layers.SpatialDropout2D(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        if self.dropout is not None:
            inputs = self.dropout(inputs, training=training)
        x1 = self.block_1x(inputs, training=training)
        x3 = self.block_3x(inputs, training=training)
        x5 = self.block_5x0(inputs, training=training)
        x5 = self.block_5x1(x5, training=training)
        mp = self.max_pooling(inputs)
        mp = self.block_max_pooling(mp, training=training)
        x = self.concatenate([x1, x3, x5, mp])
        x = self.block_compression(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
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
            'axis': self.axis,
            'rate': self.rate,
            'seed': self.seed,
            'pool_size': self.pool_size
        })
        return config
