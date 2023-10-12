import tensorflow as tf

from ..layers.embedding import FixedEmbedding
from ..layers.encoding import PositionalEmbedding2D
from ..layers.transformer import Transformer


class DETR(tf.keras.models.Model):
    def __init__(self,
                 backbone,
                 num_queries,
                 num_classes,
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
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 sparse=False,
                 normalization='layer',
                 momentum=0.99,
                 normalization_groups=32,
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 synchronized=False,
                 mode='convolution',
                 strides=None,
                 rates=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 convolution_groups=1,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.backbone = backbone
        self.convolution = tf.keras.layers.Convolution2D(
            filters=encoder_units, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, groups=convolution_groups, activation=None, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.positional_encoding = PositionalEmbedding2D(
            embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer, embeddings_constraint=embeddings_constraint,
            input_length=input_length, sparse=sparse)
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
        self.label = tf.keras.layers.Dense(
            units=num_classes + 1, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.bounding_box = tf.keras.layers.Dense(
            units=4, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.query = FixedEmbedding(
            input_dimensions=num_queries, output_dimensions=encoder_units,
            embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint)

    def call(self, inputs, training=None, mask=None):
        x = self.convolution(inputs)
        x = self.transformer([])