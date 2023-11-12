import tensorflow as tf

from .dense import DenseBlock


class MultiLayerPerceptron(tf.keras.layers.Layer):
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

        units = units if isinstance(units, (tuple, list)) else [units]
        normalization = normalization if isinstance(normalization, (tuple, list)) else [normalization] * len(units)
        if len(units) != len(normalization):
            raise ValueError(f'Number of `units` must match with number of `normalization`')
        activation = activation if isinstance(activation, (tuple, list)) else [activation]
        if len(units) != len(activation):
            raise ValueError(f'Number of `units` must match with number of `activation`')
        rate = rate if isinstance(rate, (tuple, list)) else [rate] * len(units)
        if len(units) != len(rate):
            raise ValueError(f'Number of `units` must match with number of `rate`')

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
        self.supports_masking = True

        self.mlp = [DenseBlock(
            units=_units, activation=_activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=_normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            gamma_regularizer=gamma_regularizer, moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            synchronized=synchronized, axis=axis, rate=_rate, seed=None)
            for _units, _normalization, _activation, _rate in zip(units, normalization, activation, rate)]

    def call(self, inputs, training=False, **kwargs):
        for layer in self.mlp:
            inputs = layer(inputs, training=training)
        return inputs

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
