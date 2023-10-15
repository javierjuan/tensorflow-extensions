import tensorflow as tf

from .math import MatrixMultiplication


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
        self.supports_masking = True

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
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        if self.dropout is not None:
            inputs = self.dropout(inputs, training=training)
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
