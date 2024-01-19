import keras_core as keras

from ..layers.encoding import PatchAndPositionEmbedding2D
from ..layers.mlp import MultiLayerPerceptron
from ..layers.transformer import TransformerEncoder


@keras.saving.register_keras_serializable(package='tfe.models')
class ViT(keras.Model):
    def __init__(self,
                 patch_size,
                 num_labels,
                 embedding_dimension,
                 encoder_units,
                 encoder_num_heads,
                 mlp_units,
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
                 normalization='layer',
                 momentum=0.99,
                 normalization_groups=32,
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 mode='convolution',
                 strides=None,
                 padding='same',
                 data_format=None,
                 convolution_groups=1,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size
        self.num_labels = num_labels
        self.embedding_dimension = embedding_dimension
        self.encoder_units = encoder_units
        self.encoder_num_heads = encoder_num_heads
        self.mlp_units = mlp_units
        self.use_bias = use_bias
        self.output_shape = output_shape
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
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.normalization = normalization
        self.momentum = momentum
        self.normalization_groups = normalization_groups
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.mode = mode
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.convolution_groups = convolution_groups
        self.rate = rate
        self.seed = seed

        self.is_adapted = False

        self.standardization = keras.layers.Normalization(axis=-1, dtype='float32')
        self.patch_embedding = PatchAndPositionEmbedding2D(
            size=patch_size, embedding_dimension=embedding_dimension, mode=mode, strides=strides, padding=padding,
            data_format=data_format, convolution_groups=convolution_groups, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, embeddings_constraint=embeddings_constraint, rate=rate,
            seed=seed)
        self.encoder = TransformerEncoder(
            units=encoder_units, num_heads=encoder_num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed)
        self.normalization = keras.layers.LayerNormalization(epsilon=epsilon)
        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.mlp = MultiLayerPerceptron(
            units=mlp_units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            axis=axis, rate=rate, seed=seed)
        self.posteriors = keras.layers.Dense(
            units=num_labels, activation='sigmoid', use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, dtype='float32')

    def adapt(self, data):
        self.standardization.adapt(data=data)
        self.is_adapted = True

    def build(self, input_shape):
        if not self.is_adapted:
            raise ValueError('Model must be adapted before calling `build` or `call` methods.')
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self.standardization(inputs)
        x = self.patch_embedding(x)
        x = self.encoder(x, training=training)
        x = self.normalization(x)
        x = self.pooling(x)
        x = self.mlp(x, training=training)
        return self.posteriors(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'num_labels': self.num_labels,
            'embedding_dimension': self.embedding_dimension,
            'encoder_units': self.encoder_units,
            'encoder_num_heads': self.encoder_num_heads,
            'mlp_units': self.mlp_units,
            'use_bias': self.use_bias,
            'output_shape': self.output_shape,
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
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'normalization': self.normalization,
            'momentum': self.momentum,
            'normalization_groups': self.normalization_groups,
            'moving_mean_initializer': self.moving_mean_initializer,
            'moving_variance_initializer': self.moving_variance_initializer,
            'mode': self.mode,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'convolution_groups': self.convolution_groups,
            'rate': self.rate,
            'seed': self.seed
        })
        return config
