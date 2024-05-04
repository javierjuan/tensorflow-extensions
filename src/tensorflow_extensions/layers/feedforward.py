import keras


class FeedForward(keras.layers.Layer):
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
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        units = units if isinstance(units, (tuple, list)) else [units]
        activation = activation if isinstance(activation, (tuple, list)) else [activation] * len(units)
        if len(units) != len(activation):
            raise ValueError(f'Number of `units` must match with number of `activation`')

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
        self.supports_masking = True

        self._layers = [keras.layers.Dense(
            units=_units, activation=_activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint) for _units, _activation in zip(units, activation)]

    def call(self, inputs, **kwargs):
        for layer in self._layers:
            inputs = layer(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        for layer in self._layers:
            input_shape = layer.compute_output_shape(input_shape=input_shape)
        return input_shape

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
            'bias_constraint': self.bias_constraint
        })
        return config
