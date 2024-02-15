import keras


@keras.saving.register_keras_serializable(package='tfe.layers')
class DenseBlock(keras.layers.Layer):
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
        self.units = units
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

        self.dense = keras.layers.Dense(
            units=units, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        if normalization == 'batch':
            self.normalization_ = keras.layers.BatchNormalization(
                axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer, beta_regularizer=beta_regularizer,
                moving_variance_initializer=moving_variance_initializer, gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        elif normalization == 'layer':
            self.normalization_ = keras.layers.LayerNormalization(
                axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, beta_constraint=beta_constraint,
                gamma_regularizer=gamma_regularizer, gamma_constraint=gamma_constraint)
        elif normalization == 'group':
            self.normalization_ = keras.layers.GroupNormalization(
                groups=normalization_groups, axis=axis, epsilon=epsilon, center=center, scale=scale,
                beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_constraint=beta_constraint,
                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                gamma_constraint=gamma_constraint)
        else:
            self.normalization_ = None
        self.activation_ = keras.layers.Activation(activation=activation)
        self.dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        if self.dropout is not None:
            inputs = self.dropout(inputs, training=training)
        x = self.dense(inputs)
        if self.normalization_ is not None:
            x = self.normalization_(x, training=training)
        x = self.activation_(x)
        return x

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape=input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
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
