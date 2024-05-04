import keras

from .attention import ConvolutionalAttention2D
from .residual import Residual2D


@keras.saving.register_keras_serializable(package='tfe.layers')
class ConvolutionBlock2D(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
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
                 epsilon=1e-3,
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
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
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
        self.axis = axis
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self._convolution = keras.layers.Convolution2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, groups=convolution_groups, activation=None, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        if normalization == 'batch':
            self._normalization = keras.layers.BatchNormalization(
                axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer, beta_regularizer=beta_regularizer,
                moving_variance_initializer=moving_variance_initializer, gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        elif normalization == 'layer':
            self._normalization = keras.layers.LayerNormalization(
                axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, beta_constraint=beta_constraint,
                gamma_regularizer=gamma_regularizer, gamma_constraint=gamma_constraint)
        elif normalization == 'group':
            self._normalization = keras.layers.GroupNormalization(
                groups=normalization_groups, axis=axis, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_constraint=beta_constraint,
                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                gamma_constraint=gamma_constraint)
        else:
            self._normalization = None
        self._activation = keras.layers.Activation(activation=activation)
        self._dropout = keras.layers.SpatialDropout2D(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        if self._dropout is not None:
            inputs = self._dropout(inputs, training=training)
        x = self._convolution(inputs)
        if self._normalization is not None:
            x = self._normalization(x, training=training)
        x = self._activation(x)
        return x

    def compute_output_shape(self, input_shape):
        return self._convolution.compute_output_shape(input_shape=input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
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
            'axis': self.axis,
            'rate': self.rate,
            'seed': self.seed
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class ConvolutionEncoder2DLayer(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 residual=True,
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
                 epsilon=1e-3,
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
                 pool_type='max',
                 pool_size=(2, 2),
                 attention_reduction_factor=8,
                 attention_kernel_size=(7, 7),
                 axis=-1,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.residual = residual
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
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.attention_reduction_factor = attention_reduction_factor
        self.attention_kernel_size = attention_kernel_size
        self.axis = axis
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        convolution = ConvolutionBlock2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, convolution_groups=convolution_groups, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=rate, seed=seed)
        if attention_reduction_factor is not None and attention_kernel_size is not None:
            convolution = ConvolutionalAttention2D(
                layer=convolution, reduction_factor=attention_reduction_factor, kernel_size=attention_kernel_size,
                activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, data_format=data_format,
                convolution_groups=convolution_groups)
        self.block = Residual2D(layers=convolution) if residual else convolution

        if pool_type == 'max':
            self.pooling = keras.layers.MaxPooling2D(pool_size=pool_size, strides=None, padding=padding,
                                                     data_format=data_format)
        elif pool_type == 'average':
            self.pooling = keras.layers.AveragePooling2D(pool_size=pool_size, strides=None, padding=padding,
                                                         data_format=data_format)
        else:
            raise ValueError(f'Unexpected `pooling` type {pool_type}. Must be one of `max` or `average`')

    def call(self, inputs, training=False, **kwargs):
        x = self.block(inputs, training=training)
        return self.pooling(x)

    def compute_output_shape(self, input_shape):
        return self.pooling.compute_output_shape(self.block.compute_output_shape(input_shape=input_shape))

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'residual': self.residual,
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
            'pool_type': self.pool_type,
            'pool_size': self.pool_size,
            'attention_reduction_factor': self.attention_reduction_factor,
            'attention_kernel_size': self.attention_kernel_size,
            'axis': self.axis,
            'rate': self.rate,
            'seed': self.seed
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class ConvolutionEncoder2D(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 residual=True,
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
                 epsilon=1e-3,
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
                 pool_type='max',
                 pool_size=(2, 2),
                 attention_reduction_factor=8,
                 attention_kernel_size=(7, 7),
                 axis=-1,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        filters = filters if isinstance(filters, (tuple, list)) else [filters]
        activation = activation if isinstance(activation, (tuple, list)) else [activation] * len(filters)
        normalization = normalization if isinstance(normalization, (tuple, list)) else [normalization] * len(filters)
        if len(filters) != len(normalization):
            raise ValueError(f'Number of `filters` must match with number of `normalization`')
        activation = activation if isinstance(activation, (tuple, list)) else [activation]
        if len(filters) != len(activation):
            raise ValueError(f'Number of `filters` must match with number of `activation`')
        rate = rate if isinstance(rate, (tuple, list)) else [rate] * len(filters)
        if len(filters) != len(rate):
            raise ValueError(f'Number of `filters` must match with number of `rate`')

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.residual = residual
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
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.attention_reduction_factor = attention_reduction_factor
        self.attention_kernel_size = attention_kernel_size
        self.axis = axis
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self.encoder = [ConvolutionEncoder2DLayer(
            filters=_filters, kernel_size=kernel_size, strides=strides, padding=padding, residual=residual,
            data_format=data_format, dilation_rate=dilation_rate, convolution_groups=convolution_groups,
            activation=_activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=_normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            beta_regularizer=beta_regularizer, moving_variance_initializer=moving_variance_initializer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            pool_type=pool_type, pool_size=pool_size, attention_reduction_factor=attention_reduction_factor,
            attention_kernel_size=attention_kernel_size, axis=axis, rate=_rate, seed=seed)
            for _filters, _normalization, _activation, _rate in zip(filters, normalization, activation, rate)]

    def call(self, inputs, training=False, **kwargs):
        for layer in self.encoder:
            inputs = layer(inputs, training=training)
        return inputs

    def compute_output_shape(self, input_shape):
        for layer in self.encoder:
            input_shape = layer.compute_output_shape(input_shape=input_shape)
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'residual': self.residual,
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
            'pool_type': self.pool_type,
            'pool_size': self.pool_size,
            'attention_reduction_factor': self.attention_reduction_factor,
            'attention_kernel_size': self.attention_kernel_size,
            'axis': self.axis,
            'rate': self.rate,
            'seed': self.seed
        })
        return config
