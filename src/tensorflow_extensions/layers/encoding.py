import keras_core as keras
from keras_core import ops

from .embedding import FixedEmbedding
from .math import CartesianConcatenation2D
from .patch import PatchExtractor2D


@keras.saving.register_keras_serializable(package='tfe.layers')
class PositionalEncoding1D(keras.layers.Layer):
    def __init__(self,
                 max_wavelength=10000,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_wavelength = max_wavelength
        self.supports_masking = True

        self.positions_mask = None
        self.timescales = None
        self.add = keras.layers.Add()

    def build(self, input_shape):
        embedding_dimension = input_shape[-1]
        timescales = ops.power(ops.cast(1 / self.max_wavelength, dtype=self.compute_dtype),
                               (ops.cast(2 * (ops.arange(embedding_dimension) // 2), self.compute_dtype) /
                                ops.cast(embedding_dimension, self.compute_dtype)))
        self.timescales = ops.expand_dims(timescales, axis=0)
        self.positions_mask = ops.cast(ops.arange(embedding_dimension) % 2, self.compute_dtype)
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        input_shape = ops.shape(inputs)
        positions = ops.cast(ops.arange(input_shape[-2]), dtype=self.compute_dtype)
        angles = ops.expand_dims(positions, axis=1) * self.timescales
        encoding = ops.sin(angles) * (1 - self.positions_mask) + ops.cos(angles) * self.positions_mask
        return self.add([inputs, ops.broadcast_to(encoding, shape=input_shape)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_wavelength': self.max_wavelength
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class PositionalEmbedding1D(keras.layers.Layer):
    def __init__(self,
                 sequence_length,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 input_length=None,
                 sparse=False,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sequence_length = sequence_length
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.activity_regularizer = activity_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.input_length = input_length
        self.sparse = sparse
        self.supports_masking = True

        self.embedding = None
        self.add = keras.layers.Add()

    def build(self, input_shape):
        self.embedding = keras.layers.Embedding(
            input_dim=self.sequence_length, output_dim=input_shape[-1],
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            activity_regularizer=self.activity_regularizer, embeddings_constraint=self.embeddings_constraint,
            mask_zero=False, input_length=self.input_length, sparse=self.sparse)
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        input_shape = ops.shape(inputs)
        positions = ops.cast(ops.arange(input_shape[-2]), dtype=self.compute_dtype)
        return self.add([inputs, ops.broadcast_to(self.embedding(positions), shape=input_shape)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'input_length': self.input_length,
            'sparse': self.sparse
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class PositionalEmbedding2D(keras.layers.Layer):
    def __init__(self,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.supports_masking = True

        self.row_embedding = None
        self.col_embedding = None
        self.cartesian_concatenation = CartesianConcatenation2D()
        self.add = keras.layers.Add()

    def build(self, input_shape):
        embedding_dimension = input_shape[-1] // 2
        self.row_embedding = FixedEmbedding(
            input_dim=input_shape[-3], output_dim=embedding_dimension,
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            embeddings_constraint=self.embeddings_constraint)
        self.col_embedding = FixedEmbedding(
            input_dim=input_shape[-2], output_dim=embedding_dimension,
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            embeddings_constraint=self.embeddings_constraint)
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        inputs_shape = ops.shape(inputs)
        rows_embedding = self.row_embedding(batch_size=inputs_shape[0])
        cols_embedding = self.col_embedding(batch_size=inputs_shape[0])
        embedding = self.cartesian_concatenation([rows_embedding, cols_embedding])
        embedding_shape = ops.shape(embedding)
        embedding = ops.reshape(embedding, [-1, embedding_shape[-3] * embedding_shape[-2], embedding_shape[-1]])
        inputs = ops.reshape(inputs, [-1, inputs_shape[-3] * inputs_shape[-2], inputs_shape[-1]])
        return self.add([inputs, embedding])

    def get_config(self):
        config = super().get_config()
        config.update({
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class TokenAndPositionEncoding(keras.layers.Layer):
    def __init__(self,
                 vocabulary_size,
                 embedding_dimension,
                 max_wavelength=10000,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 sparse=False,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.max_wavelength = max_wavelength
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.activity_regularizer = activity_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.sparse = sparse
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length,
            sparse=sparse)
        self.positional_encoding = PositionalEncoding1D(max_wavelength=max_wavelength)
        self.dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        x = self.token_embedding(inputs)
        x = self.positional_encoding(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocabulary_size': self.vocabulary_size,
            'embedding_dimension': self.embedding_dimension,
            'max_wavelength': self.max_wavelength,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
            'sparse': self.sparse,
            'rate': self.rate,
            'seed': self.seed
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self,
                 sequence_length,
                 vocabulary_size,
                 embedding_dimension,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 sparse=False,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.activity_regularizer = activity_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.sparse = sparse
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length,
            sparse=sparse)
        self.positional_embedding = PositionalEmbedding1D(
            sequence_length=sequence_length, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, input_length=input_length, sparse=sparse)
        self.dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        x = self.token_embedding(inputs)
        x = self.positional_embedding(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'vocabulary_size': self.vocabulary_size,
            'embedding_dimension': self.embedding_dimension,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
            'sparse': self.sparse,
            'rate': self.rate,
            'seed': self.seed
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class PatchEmbedding2D(keras.layers.Layer):
    def __init__(self,
                 size,
                 embedding_dimension,
                 mode='convolution',
                 strides=None,
                 rates=(1, 1),
                 padding='valid',
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
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.size = size
        self.embedding_dimension = embedding_dimension
        self.mode = mode
        self.strides = strides
        self.rates = rates
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
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.supports_masking = True

        self.class_token = self.add_weight(shape=[1, 1, embedding_dimension], initializer='glorot_uniform',
                                           trainable=True, name='class_token')
        if mode in ('patch', 'crop'):
            strides = size if strides is None else strides
            self.patch_extractor = PatchExtractor2D(size=size, strides=strides, rates=rates, padding=padding)
            self.dense = keras.layers.Dense(
                units=embedding_dimension, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        elif mode in ('convolution', 'conv'):
            strides = (1, 1) if strides is None else strides
            self.convolution = keras.layers.Convolution2D(
                filters=embedding_dimension, kernel_size=size, strides=strides, padding=padding,
                data_format=data_format, dilation_rate=dilation_rate, groups=convolution_groups, activation=None,
                use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint)
        else:
            raise ValueError(f'Unexpected value for `mode`: {mode}. Possible values are: `convolution`, `conv`, '
                             f'`patch` or `crop`.')
        self.add = keras.layers.Add()
        self.embedding = None

    def build(self, input_shape):
        # TODO: Not sure input_dim=input_shape[-2]
        self.embedding = FixedEmbedding(
            input_dim=input_shape[-2], output_dim=self.embedding_dimension,
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            embeddings_constraint=self.embeddings_constraint)
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        # TODO: Class Token
        if self.mode in ('patch', 'crop'):
            patches = self.patch_extractor(inputs)
            patches = self.dense(patches)
        else:
            patches = self.convolution(inputs)
        patches = ops.reshape(patches, shape=[-1, patches.shape[1] * patches.shape[2], patches.shape[3]])
        return self.add([patches, self.embedding(None)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size,
            'embedding_dimension': self.embedding_dimension,
            'mode': self.mode,
            'strides': self.strides,
            'rates': self.rates,
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
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint
        })
        return config
