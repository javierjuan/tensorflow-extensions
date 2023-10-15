import tensorflow as tf

from .model import Model
from ..layers.encoding import TokenAndPositionEncoding, TokenAndPositionEmbedding
from ..layers.transformer import Transformer


class Seq2SeqTransformer(Model):
    def __init__(self,
                 input_vocabulary_size,
                 output_vocabulary_size,
                 sequence_length,
                 embedding_dimension,
                 encoder_units,
                 encoder_num_heads,
                 decoder_units=None,
                 decoder_num_heads=None,
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
                 input_length=None,
                 sparse=False,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if positional == 'encoding':
            self.input_embedding = TokenAndPositionEncoding(
                vocabulary_size=input_vocabulary_size, embedding_dimension=embedding_dimension,
                max_wavelength=max_wavelength, embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length,
                sparse=sparse, rate=rate, seed=seed)
        else:
            self.input_embedding = TokenAndPositionEmbedding(
                sequence_length=sequence_length, vocabulary_size=input_vocabulary_size,
                embedding_dimension=embedding_dimension, embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length,
                sparse=sparse, rate=rate, seed=seed)
        if positional == 'encoding':
            self.output_embedding = TokenAndPositionEncoding(
                vocabulary_size=output_vocabulary_size, embedding_dimension=embedding_dimension,
                max_wavelength=max_wavelength, embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length,
                sparse=sparse, rate=rate, seed=seed)
        else:
            self.output_embedding = TokenAndPositionEmbedding(
                sequence_length=sequence_length, vocabulary_size=output_vocabulary_size,
                embedding_dimension=embedding_dimension, embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length,
                sparse=sparse, rate=rate, seed=seed)
        self.transformer = Transformer(
            encoder_units=encoder_units, encoder_num_heads=encoder_num_heads, decoder_units=decoder_units,
            decoder_num_heads=decoder_num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed)
        self.posteriors = tf.keras.layers.Dense(
            units=output_vocabulary_size, activation='softmax', use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, dtype=tf.float32)

    def call(self, inputs, training=False, **kwargs):
        inputs, outputs = inputs
        input_embedding = self.input_embedding(inputs, training=training)
        output_embedding = self.output_embedding(outputs, training=training)
        outputs = self.transformer([input_embedding, output_embedding], training=training)
        return self.posteriors(outputs)
