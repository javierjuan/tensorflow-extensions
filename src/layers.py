import tensorflow as tf


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self,
                 units,
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
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self._activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self._normalization = normalization
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
        self.rate = rate
        self.seed = seed

        self.dense = tf.keras.layers.Dense(
            units=units, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        if normalization == 'batch':
            self.normalization = tf.keras.layers.BatchNormalization(
                axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer, beta_regularizer=beta_regularizer,
                moving_variance_initializer=moving_variance_initializer, gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, synchronized=synchronized)
        elif normalization == 'layer':
            self.normalization = tf.keras.layers.LayerNormalization(
                axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, beta_constraint=beta_constraint,
                gamma_regularizer=gamma_regularizer, gamma_constraint=gamma_constraint)
        elif normalization == 'group':
            self.normalization = tf.keras.layers.GroupNormalization(
                groups=normalization_groups, axis=axis, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_constraint=beta_constraint,
                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                gamma_constraint=gamma_constraint)
        else:
            self.normalization = None
        self.activation = tf.keras.layers.Activation(activation=activation)
        self.dropout = tf.keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        if self.dropout is not None:
            inputs = self.dropout(inputs, training=training)
        x = self.dense(inputs)
        if self.normalization is not None:
            x = self.normalization(x, training=training)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self._activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'normalization': self._normalization,
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
            'rate': self.rate,
            'seed': self.seed
        })
        return config


class ConvolutionBlock2D(tf.keras.layers.Layer):
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
        self._activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self._normalization = normalization
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
        self.rate = rate
        self.seed = seed

        self.convolution = tf.keras.layers.Convolution2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, groups=convolution_groups, activation=None, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        if normalization == 'batch':
            self.normalization = tf.keras.layers.BatchNormalization(
                axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer, beta_regularizer=beta_regularizer,
                moving_variance_initializer=moving_variance_initializer, gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, synchronized=synchronized)
        elif normalization == 'layer':
            self.normalization = tf.keras.layers.LayerNormalization(
                axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, beta_constraint=beta_constraint,
                gamma_regularizer=gamma_regularizer, gamma_constraint=gamma_constraint)
        elif normalization == 'group':
            self.normalization = tf.keras.layers.GroupNormalization(
                groups=normalization_groups, axis=axis, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_constraint=beta_constraint,
                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                gamma_constraint=gamma_constraint)
        else:
            self.normalization = None
        self.activation = tf.keras.layers.Activation(activation=activation)
        self.dropout = tf.keras.layers.SpatialDropout2D(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        if self.dropout is not None:
            inputs = self.dropout(inputs, training=training)
        x = self.convolution(inputs)
        if self.normalization is not None:
            x = self.normalization(x, training=training)
        x = self.activation(x)
        return x

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
            'activation': self._activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'normalization': self._normalization,
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
            'rate': self.rate,
            'seed': self.seed
        })
        return config


class InceptionBlock2D(tf.keras.layers.Layer):
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
                 synchronized=False,
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
        self.synchronized = synchronized
        self.axis = axis
        self.rate = rate
        self.seed = seed
        self.pool_size = pool_size

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
            synchronized=synchronized, axis=axis, rate=None, seed=None)
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
            synchronized=synchronized, axis=axis, rate=None, seed=None)
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
            synchronized=synchronized, axis=axis, rate=None, seed=None)
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
            synchronized=synchronized, axis=axis, rate=None, seed=None)
        self.max_pooling = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
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
            synchronized=synchronized, axis=axis, rate=None, seed=None)
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
            synchronized=synchronized, axis=axis, rate=None, seed=None)
        self.concatenate = tf.keras.layers.Concatenate(axis=axis)
        self.dropout = tf.keras.layers.SpatialDropout2D(rate=rate, seed=seed) if rate is not None else None

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
            'synchronized': self.synchronized,
            'axis': self.axis,
            'rate': self.rate,
            'seed': self.seed,
            'pool_size': self.pool_size
        })
        return config


class NonLocalBlock2D(tf.keras.layers.Layer):
    def __init__(self,
                 mode,
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
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.mode = mode
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
        self._normalization = normalization
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
        self.rate = rate
        self.seed = seed

        self.g = None
        self.theta = None
        self.phi = None
        self.convolution = None
        if normalization == 'batch':
            self.normalization = tf.keras.layers.BatchNormalization(
                axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer, beta_regularizer=beta_regularizer,
                moving_variance_initializer=moving_variance_initializer, gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, synchronized=synchronized)
        elif normalization == 'layer':
            self.normalization = tf.keras.layers.LayerNormalization(
                axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, beta_constraint=beta_constraint,
                gamma_regularizer=gamma_regularizer, gamma_constraint=gamma_constraint)
        elif normalization == 'group':
            self.normalization = tf.keras.layers.GroupNormalization(
                groups=normalization_groups, axis=axis, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_constraint=beta_constraint,
                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                gamma_constraint=gamma_constraint)
        else:
            self.normalization = None
        self.flatten = None
        self.expand = None
        self.multiplication = MatrixMultiplication()
        self.softmax = tf.keras.layers.Softmax()
        self.dropout = tf.keras.layers.SpatialDropout2D(rate=rate, seed=seed) if rate is not None else None

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        filters = int(max(1, input_shape[-1] // 2))
        self.convolution = tf.keras.layers.Convolution2D(
            filters=filters, kernel_size=(1, 1), strides=self.strides, padding=self.padding,
            data_format=self.data_format, dilation_rate=self.dilation_rate, groups=self.convolution_groups,
            activation=None, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint)
        self.g = tf.keras.layers.Convolution2D(
            filters=filters, kernel_size=(1, 1), strides=self.strides, padding=self.padding,
            data_format=self.data_format, dilation_rate=self.dilation_rate, groups=self.convolution_groups,
            activation=None, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint)
        if self.mode == 'embedding' or self.mode == 'dot' or self.mode == 'concat':
            self.theta = tf.keras.layers.Convolution2D(
                filters=filters, kernel_size=(1, 1), strides=self.strides, padding=self.padding,
                data_format=self.data_format, dilation_rate=self.dilation_rate, groups=self.convolution_groups,
                activation=None, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint)
            self.phi = tf.keras.layers.Convolution2D(
                filters=filters, kernel_size=(1, 1), strides=self.strides, padding=self.padding,
                data_format=self.data_format, dilation_rate=self.dilation_rate, groups=self.convolution_groups,
                activation=None, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint)
        self.flatten = tf.keras.layers.Reshape([-1, filters])
        self.expand = tf.keras.layers.Reshape(input_shape[1:-1] + filters)

    def call(self, inputs, training=False, **kwargs):
        x, y = inputs, inputs
        g = self.flatten(self.g(inputs))
        if self.mode == 'embedding' or self.mode == 'dot':
            x, y = self.theta(inputs), self.phi(inputs)
        elif self.mode == 'concatenation':
            pass
        else:
            raise ValueError(f'Unexpected `mode` value: {self.mode}')
        f = self.multiplication(a=self.flatten(x), b=self.flatten(y), transpose_b=True)
        if self.mode == 'gaussian' or self.mode == 'embedding':
            f = self.softmax(f)
        elif self.mode == 'dot' or self.mode == 'concatenate':
            f /= f.shape[1]
        z = self.expand(self.multiplication(a=f, b=g))
        z = self.convolution(z)
        if self.normalization is not None:
            z = self.normalization(z)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({
            'mode': self.mode,
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
            'bias_constraint': self.bias_constraint,
            'normalization': self._normalization,
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
            'rate': self.rate,
            'seed': self.seed
        })
        return config


class ResidualBlock2D(tf.keras.layers.Layer):
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
        self.layer = layer if isinstance(layer, tf.keras.layers.Layer) else tf.keras.layers.deserialize(layer)
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
        self.add = tf.keras.layers.Add()
        self.adapt_input = None

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        input_channels = input_shape[-1]
        layer_channels = self.layer.compute_output_shape(input_shape=input_shape)[-1]
        if input_channels != layer_channels:
            self.adapt_input = tf.keras.layers.Convolution2D(
                filters=layer_channels, kernel_size=(1, 1), padding='same', activation=None)

    def call(self, inputs, training=False, **kwargs):
        x = self.layer(inputs, training=training)
        if self.attention_block is not None:
            x = self.attention_block(x, training=training)
        x = self.add([x, self.adapt_input(inputs, training=training) if self.adapt_input is not None else inputs])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer': tf.keras.layers.serialize(self.layer),
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


class TransformerAttention(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_dimension,
                 num_heads,
                 dropout=0.0,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self._output_shape = output_shape
        self.attention_axes = attention_axes
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.attention_scores = None
        self.supports_masking = True

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dimension, value_dim=None, dropout=dropout, use_bias=use_bias,
            output_shape=output_shape, attention_axes=attention_axes, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.normalization = tf.keras.layers.LayerNormalization(
            axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        if hasattr(self.attention, '_build_from_signature'):
            self.attention._build_from_signature(query=input_shape, value=input_shape)

    def call(self, inputs, context=None, training=False, mask=None, return_attention_scores=False,
             use_causal_mask=False, **kwargs):
        x = self.normalization(inputs)
        context = x if context is None else context
        if return_attention_scores:
            x, attention_scores = self.attention(query=x, value=context, key=context, training=training,
                                                 attention_mask=mask, use_causal_mask=use_causal_mask,
                                                 return_attention_scores=return_attention_scores)
            self.attention_scores = attention_scores
        else:
            x = self.attention(query=x, value=context, key=context, training=training, attention_mask=mask,
                               use_causal_mask=use_causal_mask, return_attention_scores=return_attention_scores)
        return self.add([x, inputs])

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dimension': self.embedding_dimension,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'use_bias': self.use_bias,
            'output_shape': self._output_shape,
            'attention_axes': self.attention_axes,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint
        })


class TransformerFeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_dimension,
                 dense_dimension,
                 activation='mish',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
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
        self.embedding_dimension = embedding_dimension
        self.dense_dimension = dense_dimension
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.axis = axis
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self.dense_1 = self.dense = tf.keras.layers.Dense(
            units=dense_dimension, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.dense_2 = self.dense = tf.keras.layers.Dense(
            units=embedding_dimension, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.normalization = tf.keras.layers.LayerNormalization(
            axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.dropout = tf.keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=False, **kwargs):
        x = self.normalization(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return self.add([x, inputs])

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dimension': self.embedding_dimension,
            'dense_dimension': self.dense_dimension,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint,
            'axis': self.axis,
            'rate': self.rate,
            'seed': self.seed
        })


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_dimension,
                 dense_dimension,
                 num_heads,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activation='mish',
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dimension = embedding_dimension
        self.dense_dimension = dense_dimension
        self.num_heads = num_heads
        self.use_bias = use_bias
        self._output_shape = output_shape
        self.attention_axes = attention_axes
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.activation = activation
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self.self_attention = TransformerAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads, dropout=rate, use_bias=use_bias,
            output_shape=output_shape, attention_axes=attention_axes, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, axis=axis, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.feed_forward = TransformerFeedForward(
            embedding_dimension=embedding_dimension, dense_dimension=dense_dimension, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=rate, seed=seed)

    def call(self, inputs, training=False, mask=None, return_attention_scores=False, **kwargs):
        x = self.self_attention(inputs, context=None, training=training, mask=mask, use_causal_mask=False,
                                return_attention_scores=return_attention_scores)
        x = self.feed_forward(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dimension': self.embedding_dimension,
            'dense_dimension': self.dense_dimension,
            'num_heads': self.num_heads,
            'use_bias': self.use_bias,
            'output_shape': self._output_shape,
            'attention_axes': self.attention_axes,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'activation': self.activation,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint
        })
        return config


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_dimension,
                 dense_dimension,
                 num_heads,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activation='mish',
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dimension = embedding_dimension
        self.dense_dimension = dense_dimension
        self.num_heads = num_heads
        self.use_bias = use_bias
        self._output_shape = output_shape
        self.attention_axes = attention_axes
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.activation = activation
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self.self_attention = TransformerAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads, dropout=rate, use_bias=use_bias,
            output_shape=output_shape, attention_axes=attention_axes, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, axis=axis, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.cross_attention = TransformerAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads, dropout=rate, use_bias=use_bias,
            output_shape=output_shape, attention_axes=attention_axes, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, axis=axis, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.feed_forward = TransformerFeedForward(
            embedding_dimension=embedding_dimension, dense_dimension=dense_dimension, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=rate, seed=seed)
        self.attention_scores = None

    def call(self, inputs, context=None, training=False, mask=None, return_attention_scores=False, **kwargs):
        x = self.self_attention(inputs, context=None, training=training, mask=mask, use_causal_mask=True,
                                return_attention_scores=return_attention_scores)
        x = self.cross_attention(x, context=context, training=training, mask=mask, use_causal_mask=False,
                                 return_attention_scores=return_attention_scores)
        self.attention_scores = self.cross_attention.attention_scores
        x = self.feed_forward(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dimension': self.embedding_dimension,
            'dense_dimension': self.dense_dimension,
            'num_heads': self.num_heads,
            'use_bias': self.use_bias,
            'output_shape': self._output_shape,
            'attention_axes': self.attention_axes,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'activation': self.activation,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint
        })
        return config


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocabulary_size,
                 embedding_dimension,
                 max_input_length=1024,
                 positional='embedding',
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 sparse=False,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.positional = positional
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.activity_regularizer = activity_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.sparse = sparse
        self.rate = rate
        self.seed = seed

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length,
            sparse=sparse)
        self.positional_encoding = PositionalEncoding(
            mode=positional, max_input_length=max_input_length, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, input_length=input_length, sparse=sparse)
        self.dropout = tf.keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def compute_mask(self, inputs, mask=None):
        return self.token_embedding.compute_mask(mask)

    def call(self, inputs, training=False, **kwargs):
        x = self.token_embedding(inputs)
        x = self.positional_encoding(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocabulary_size': self.vocabulary_size,
            'embedding_dimension': self.embedding_dimension,
            'max_input_length': self.max_input_length,
            'positional': self.positional,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
            'sparse': self.sparse
        })


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,
                 mode='embedding',
                 max_input_length=1024,
                 max_wavelength=10000,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 input_length=None,
                 sparse=False,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.mode = mode
        self.max_input_length = max_input_length
        self.max_wavelength = max_wavelength
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.activity_regularizer = activity_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.input_length = input_length
        self.sparse = sparse
        self.supports_masking = True
        self.embedding = None
        self.positions_mask = None
        self.timescales = None
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        embedding_dimension = input_shape[-1]
        if self.mode == 'encoding':
            self.timescales = tf.math.pow(
                tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype),
                (tf.cast(2 * (tf.range(embedding_dimension) // 2), self.compute_dtype) /
                 tf.cast(embedding_dimension, self.compute_dtype)))
            self.positions_mask = tf.cast(tf.range(embedding_dimension) % 2, self.compute_dtype)
        elif self.mode == 'embedding':
            self.embedding = tf.keras.layers.Embedding(
                input_dim=self.max_input_length, output_dim=embedding_dimension,
                embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
                activity_regularizer=self.activity_regularizer, embeddings_constraint=self.embeddings_constraint,
                mask_zero=False, input_length=self.input_length, sparse=self.sparse)
        else:
            raise ValueError(f'Unexpected `mode`: {self.mode}. Possible values are: `encoding` or `embedding`')

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        positions = tf.cast(tf.range(start=0, limit=input_shape[-2], delta=1), self.compute_dtype)
        if self.mode == 'encoding':
            angles = tf.expand_dims(positions, axis=1) * tf.expand_dims(self.timescales, axis=0)
            encoding = (tf.math.sin(angles) * (1 - self.positions_mask) + tf.math.cos(angles) * self.positions_mask)
            return self.add([inputs, tf.broadcast_to(encoding, shape=input_shape)])
        else:
            return self.add([inputs, tf.broadcast_to(self.embedding(positions), shape=input_shape)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'mode': self.mode,
            'max_input_length': self.max_input_length,
            'max_wavelength': self.max_wavelength,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'input_length': self.input_length,
            'sparse': self.sparse
        })


class PatchEncoding2D(tf.keras.layers.Layer):
    def __init__(self,
                 size,
                 embedding_dimension,
                 mode='convolution',
                 strides=None,
                 rates=(1, 1),
                 padding='valid',
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
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 sparse=False,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        strides = size if strides is None else strides
        self.size = size
        self.embedding_dimension = embedding_dimension
        self.mode = mode
        self.strides = strides
        self.rates = rates
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
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.sparse = sparse

        self.class_token = self.add_weight(shape=[1, 1, embedding_dimension], name='class_token', trainable=True)
        if mode in ('patch', 'crop'):
            self.dense = tf.keras.layers.Dense(
                units=embedding_dimension, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        elif mode in ('convolution', 'conv'):
            self.convolution = tf.keras.layers.Convolution2D(
                filters=embedding_dimension, kernel_size=size, strides=strides, padding=padding,
                data_format=data_format, dilation_rate=dilation_rate, groups=convolution_groups, activation=None,
                use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint)
        else:
            raise ValueError(f'Unexpected value for `mode`: {mode}. Possible values are: `convolution`, `conv`, '
                             f'`patch` or `crop`.')
        self.add = tf.keras.layers.Add()
        self.embedding = None
        self.positions = None

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_shape[-2], output_dim=self.embedding_dimension,
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            activity_regularizer=self.activity_regularizer, embeddings_constraint=self.embeddings_constraint,
            mask_zero=self.mask_zero, input_length=self.input_length, sparse=self.sparse)
        self.positions = tf.expand_dims(tf.range(start=0, limit=input_shape[1], delta=1), axis=0)

    def call(self, inputs, **kwargs):
        if self.mode in ('patch', 'crop'):
            patches = tf.image.extract_patches(images=inputs, sizes=[1, *self.size, 1], strides=[1, *self.strides, 1],
                                               rates=[1, *self.rates, 1], padding=self.padding.upper())
            patches = self.dense(patches)
        else:
            patches = self.convolution(inputs)
        patches = tf.reshape(patches, shape=[-1, patches.shape[1] * patches.shape[2], patches.shape[3]])
        return self.add([patches, self.embedding(self.positions)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size,
            'embedding_dimension': self.embedding_dimension,
            'mode': self.mode,
            'strides': self.strides,
            'rates': self.rates,
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
            'bias_constraint': self.bias_constraint,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
            'sparse': self.sparse
        })


class PatchExtractor2D(tf.keras.layers.Layer):
    def __init__(self,
                 size,
                 stride=None,
                 rate=(1, 1),
                 padding='valid',
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if len(size) != 2:
            raise ValueError(f'`size` must be an array-like of size 2. Got: {size}')
        if stride is not None and len(stride) != 2:
            raise ValueError(f'`stride` must be an array-like of size 2. Got: {stride}')
        if rate is not None and len(rate) != 2:
            raise ValueError(f'`rate` must be an array-like of size 2. Got: {rate}')

        self.size = [1, *size, 1]
        self.stride = self.size if stride is None else [1, *stride, 1]
        self.rate = [1, 1, 1, 1] if rate is None else [1, *rate, 1]
        self.padding = padding

    def call(self, inputs, **kwargs):
        patches = tf.image.extract_patches(images=inputs, sizes=self.size, strides=self.stride, rates=self.rate,
                                           padding=self.padding.upper())
        return tf.reshape(patches, shape=[-1, patches.shape[1] * patches.shape[2], patches.shape[3]])

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size[1:-1],
            'stride': self.stride[1:-1],
            'rate': self.rate[1:-1],
            'padding': self.padding
        })
        return config


class ChannelAveragePooling(tf.keras.layers.Layer):
    def __init__(self,
                 keepdims=False,
                 data_format=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.keepdims = keepdims
        self.data_format = tf.keras.backend.image_data_format().lower() if data_format is None else data_format.lower()
        if self.data_format not in ('channels_first', 'channels_last'):
            raise ValueError('The `data_format` argument must be one of "channels_first", "channels_last". '
                             f'Received: {self.data_format}')

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            return tf.math.reduce_mean(inputs, axis=-1, keepdims=self.keepdims)
        else:
            return tf.math.reduce_mean(inputs, axis=1, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({
            'keepdims': self.keepdims,
            'data_format': self.data_format
        })
        return config


class ChannelMaxPooling(tf.keras.layers.Layer):
    def __init__(self,
                 keepdims=False,
                 data_format=None,
                 name=None,
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.keepdims = keepdims
        self.data_format = tf.keras.backend.image_data_format().lower() if data_format is None else data_format.lower()
        if self.data_format not in ('channels_first', 'channels_last'):
            raise ValueError('The `data_format` argument must be one of "channels_first", "channels_last". '
                             f'Received: {self.data_format}')

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            return tf.math.reduce_max(inputs, axis=-1, keepdims=self.keepdims)
        else:
            return tf.math.reduce_max(inputs, axis=1, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({
            'keepdims': self.keepdims,
            'data_format': self.data_format
        })
        return config


class ExpandDimensions(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis
        })
        return config


class MathReduce(tf.keras.layers.Layer):
    def __init__(self,
                 reduce_mode,
                 axis=-1,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduce_mode = reduce_mode
        self.axis = axis

        if reduce_mode == 'min':
            self.reduce_operation = tf.math.reduce_min
        elif reduce_mode == 'max':
            self.reduce_operation = tf.math.reduce_max
        elif reduce_mode == 'mean':
            self.reduce_operation = tf.math.reduce_mean
        elif reduce_mode == 'sum':
            self.reduce_operation = tf.math.reduce_sum
        elif reduce_mode == 'prod':
            self.reduce_operation = tf.math.reduce_prod
        else:
            raise ValueError(f'Unexpected operation {reduce_mode}')

    def call(self, inputs, **kwargs):
        return self.reduce_operation(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduce_mode': self.reduce_mode,
            'axis': self.axis
        })
        return config


class MatrixMultiplication(tf.keras.layers.Layer):
    def call(self, a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, **kwargs):
        return tf.linalg.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a,
                                adjoint_b=adjoint_b)
