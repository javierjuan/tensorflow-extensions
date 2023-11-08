from collections.abc import Sequence

import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class TransformerAttention(tf.keras.layers.Layer):
    def __init__(self,
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
        self.num_heads = num_heads
        self.dropout = 0.0 if dropout is None else dropout
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

        self.attention = None
        self.normalization = tf.keras.layers.LayerNormalization(
            axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        embedding_dimension = input_shape[-1]
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=embedding_dimension // self.num_heads, value_dim=None,
            dropout=self.dropout, use_bias=self.use_bias, output_shape=self._output_shape,
            attention_axes=self.attention_axes, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint)
        if hasattr(self.attention, "_build_from_signature"):
            self.attention._build_from_signature(query=input_shape, value=input_shape)
        else:
            self.attention.build(query_shape=input_shape, value_shape=input_shape)
        super().build(input_shape=input_shape)

    def call(self, inputs, context=None, training=False, use_causal_mask=False, **kwargs):
        x = self.normalization(inputs)
        context = x if context is None else context
        x, self.attention_scores = self.attention(query=x, value=context, key=context, return_attention_scores=True,
                                                  training=training, use_causal_mask=use_causal_mask)
        return self.add([x, inputs])

    def get_config(self):
        config = super().get_config()
        config.update({
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
        return config


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class TransformerFeedForward(tf.keras.layers.Layer):
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

        self.dense_input = tf.keras.layers.Dense(
            units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.dense_output = None
        self.normalization = tf.keras.layers.LayerNormalization(
            axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.dropout = tf.keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        embedding_dimension = input_shape[-1]
        self.dense_output = tf.keras.layers.Dense(
            units=embedding_dimension, activation=None, use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer, kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self.normalization(inputs)
        x = self.dense_input(x)
        x = self.dense_output(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return self.add([x, inputs])

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
        return config


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 units,
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
        self.units = units
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
            num_heads=num_heads, dropout=rate, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, axis=axis, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.feed_forward = TransformerFeedForward(
            units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=rate, seed=seed)

    def call(self, inputs, training=False, **kwargs):
        x = self.self_attention(inputs, training=training)
        x = self.feed_forward(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
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


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 units,
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
        self.units = units
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
            num_heads=num_heads, dropout=rate, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, axis=axis, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.cross_attention = TransformerAttention(
            num_heads=num_heads, dropout=rate, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, axis=axis, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self.feed_forward = TransformerFeedForward(
            units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=rate, seed=seed)

    def call(self, inputs, context=None, training=False, **kwargs):
        x = self.self_attention(inputs, context=None, training=training, use_causal_mask=True)
        x = self.cross_attention(x, context=context, training=training, use_causal_mask=False)
        x = self.feed_forward(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
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


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 units,
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

        units = units if isinstance(units, Sequence) else [units]
        num_heads = num_heads if isinstance(num_heads, Sequence) else [num_heads] * len(units)
        if len(units) != len(num_heads):
            raise ValueError(f'Number of `units` must match with number of `num_heads`')

        self.units = units
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

        self.encoder = [TransformerEncoderLayer(
            units=_units, num_heads=_num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed) for _units, _num_heads in zip(units, num_heads)]

    def call(self, inputs, training=False, **kwargs):
        for layer in self.encoder:
            inputs = layer(inputs, training=training)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
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


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 units,
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

        units = units if isinstance(units, Sequence) else [units]
        num_heads = num_heads if isinstance(num_heads, Sequence) else [num_heads] * len(units)
        if len(units) != len(num_heads):
            raise ValueError(f'Number of `units` must match with number of `num_heads`')

        self.units = units
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

        self.decoder = [TransformerDecoderLayer(
            units=_units, num_heads=_num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=activity_regularizer, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed) for _units, _num_heads in zip(units, num_heads)]

    def call(self, inputs, context=None, training=False, **kwargs):
        for layer in self.decoder:
            inputs = layer(inputs, context=context, training=training)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
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


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class Transformer(tf.keras.layers.Layer):
    def __init__(self,
                 encoder_units,
                 encoder_num_heads,
                 decoder_units=None,
                 decoder_num_heads=None,
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

        decoder_units = encoder_units if decoder_units is None else decoder_units
        decoder_num_heads = encoder_num_heads if decoder_num_heads is None else decoder_num_heads

        self.encoder_units = encoder_units
        self.encoder_num_heads = encoder_num_heads
        self.decoder_units = decoder_units
        self.decoder_num_heads = decoder_num_heads
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

        self.encoder = TransformerEncoder(
            units=encoder_units, num_heads=encoder_num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed)
        self.decoder = TransformerDecoder(
            units=decoder_units, num_heads=decoder_num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=activity_regularizer, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed)

    def call(self, inputs, training=False, **kwargs):
        inputs, outputs = inputs
        inputs = self.encoder(inputs, training=training)
        outputs = self.decoder(outputs, context=inputs, training=training)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder_units': self.encoder_units,
            'decoder_units': self.decoder_units,
            'encoder_num_heads': self.encoder_num_heads,
            'decoder_num_heads': self.decoder_num_heads,
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
            'gamma_constraint': self.gamma_constraint,
            'rate': self.rate,
            'seed': self.seed
        })
        return config
