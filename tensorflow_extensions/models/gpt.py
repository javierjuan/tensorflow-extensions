import keras

from ..layers.encoding import TokenAndPositionEncoding, TokenAndPositionEmbedding
from ..layers.transformer import TransformerDecoder


@keras.saving.register_keras_serializable(package='tfe.models')
class GPT(keras.Model):
    def __init__(self,
                 vocabulary_size,
                 embedding_dimension,
                 units,
                 num_heads,
                 positional='embedding',
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
                 mask_zero=False,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if positional == 'encoding':
            self._embedding = TokenAndPositionEncoding(
                vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension,
                max_wavelength=max_wavelength, embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, rate=rate, seed=seed)
        else:
            self._embedding = TokenAndPositionEmbedding(
                vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension,
                embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
                activity_regularizer=activity_regularizer, embeddings_constraint=embeddings_constraint,
                mask_zero=mask_zero, rate=rate, seed=seed)
        self.decoder = TransformerDecoder(
            units=units, num_heads=num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed)
        self._posteriors = keras.layers.Dense(
            units=vocabulary_size, activation='softmax', use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, dtype='float32')

    def call(self, inputs, training=False, **kwargs):
        x = self._embedding(inputs, training=training)
        x = self.decoder(x, training=training)
        return self._posteriors(x)
