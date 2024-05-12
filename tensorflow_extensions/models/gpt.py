import keras

from ..layers.encoding import TokenAndPositionEncoding, TokenAndPositionEmbedding
from ..layers.transformer import TransformerDecoder


@keras.saving.register_keras_serializable(package='tfe.models')
class GPT(keras.Model):
    def __init__(self,
                 units,
                 num_heads,
                 sequence_length,
                 vocabulary_size,
                 embedding_dimension,
                 position='embedding',
                 max_wavelength=10000,
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
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=True,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if position == 'encoding':
            self._embedding = TokenAndPositionEncoding(
                vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension,
                max_wavelength=max_wavelength, embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, rate=rate, seed=seed)
        else:
            self._embedding = TokenAndPositionEmbedding(
                vocabulary_size=vocabulary_size, sequence_length=sequence_length,
                embedding_dimension=embedding_dimension, embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, rate=rate, seed=seed)
        self._decoder = TransformerDecoder(
            units=units, num_heads=num_heads, embedding_dimension=embedding_dimension, use_bias=use_bias,
            output_shape=output_shape, attention_axes=attention_axes, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed)
        self._normalization = keras.layers.LayerNormalization(
            axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)
        self._posteriors = keras.layers.Dense(
            units=vocabulary_size, activation='softmax', use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, dtype='float32')

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            return
        if input_shape and not (isinstance(input_shape[0], int) or input_shape[0] is None):
            return
        x = keras.Input(batch_shape=input_shape)
        for layer in self.layers:
            x = layer(x)
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self._embedding(inputs, training=training)
        x = self._decoder(x, training=training)
        x = self._normalization(x)
        x = self._posteriors(x)
        return x
