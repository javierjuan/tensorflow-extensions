import keras
from keras import ops

from .embedding import FixedEmbedding
from .patch import PatchExtractor2D


@keras.saving.register_keras_serializable(package='tfe.layers')
class PositionEncoding1D(keras.layers.Layer):
    def __init__(self,
                 embedding_dimension,
                 max_wavelength=10000,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dimension = embedding_dimension
        self.max_wavelength = max_wavelength
        self.supports_masking = True

        timescales = ops.power(ops.cast(1 / self.max_wavelength, dtype=self.compute_dtype),
                               (ops.cast(2 * (ops.arange(embedding_dimension) // 2), self.compute_dtype) /
                                ops.cast(embedding_dimension, self.compute_dtype)))
        self._timescales = ops.expand_dims(timescales, axis=0)
        self._positions_mask = ops.cast(ops.arange(embedding_dimension) % 2, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.embedding_dimension,)

    def call(self, inputs, **kwargs):
        positions = ops.arange(inputs.shape[-1], dtype=self.compute_dtype)
        angles = ops.expand_dims(positions, axis=1) * self._timescales
        encoding = ops.sin(angles) * (1 - self._positions_mask) + ops.cos(angles) * self._positions_mask
        return ops.tile(encoding, repeats=(inputs.shape[0], 1, 1))

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dimension': self.embedding_dimension,
            'max_wavelength': self.max_wavelength
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class PositionEmbedding1D(FixedEmbedding):
    def __init__(self,
                 sequence_length,
                 embedding_dimension,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 name=None,
                 **kwargs):
        # Since we are using FixedEmbedding, we are assuming that inputs are dense tensors always padded
        # (batch_size, sequence_length) shape
        super().__init__(input_dim=sequence_length, output_dim=embedding_dimension,
                         embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
                         embeddings_constraint=embeddings_constraint, name=name, **kwargs)
        self.sequence_length = sequence_length
        self.embedding_dimension = embedding_dimension

    def get_config(self):
        config = super().get_config()
        config.pop('input_dim')
        config.pop('output_dim')
        config.update({
            'sequence_length': self.sequence_length,
            'embedding_dimension': self.embedding_dimension,
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class PositionEmbedding2D(FixedEmbedding):
    def __init__(self,
                 size,
                 embedding_dimension,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 name=None,
                 **kwargs):
        super().__init__(input_dim=size[1] * size[2], output_dim=embedding_dimension,
                         embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
                         embeddings_constraint=embeddings_constraint, name=name, **kwargs)
        self.size = size
        self.embedding_dimension = embedding_dimension

    def get_config(self):
        config = super().get_config()
        config.pop('input_dim')
        config.pop('output_dim')
        config.update({
            'size': self.size,
            'embedding_dimension': self.embedding_dimension,
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
                 embeddings_constraint=None,
                 mask_zero=False,
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
        self.embeddings_constraint = embeddings_constraint
        self.mask_zero = mask_zero
        self.rate = rate
        self.seed = seed

        self._token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero)
        self._position_encoding = PositionEncoding1D(embedding_dimension=embedding_dimension,
                                                     max_wavelength=max_wavelength)
        self._add = keras.layers.Add()
        self._dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None
        self.supports_masking = self._token_embedding.supports_masking

    def call(self, inputs, training=False, **kwargs):
        x = self._token_embedding(inputs)
        y = self._position_encoding(inputs)
        z = self._add([x, y])
        if self._dropout is not None:
            z = self._dropout(z, training=training)
        return z

    def compute_mask(self, inputs, mask=None):
        return self._token_embedding.compute_mask(inputs=inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        return self._token_embedding.compute_output_shape(input_shape=input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocabulary_size': self.vocabulary_size,
            'embedding_dimension': self.embedding_dimension,
            'max_wavelength': self.max_wavelength,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'mask_zero': self.mask_zero,
            'rate': self.rate,
            'seed': self.seed
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self,
                 vocabulary_size,
                 sequence_length,
                 embedding_dimension,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.embedding_dimension = embedding_dimension
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.mask_zero = mask_zero
        self.rate = rate
        self.seed = seed

        self._token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero)
        self._position_embedding = PositionEmbedding1D(
            sequence_length=sequence_length, embedding_dimension=embedding_dimension,
            embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint)
        self._add = keras.layers.Add()
        self._dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None
        self.supports_masking = self._token_embedding.supports_masking

    def call(self, inputs, training=False, **kwargs):
        x = self._token_embedding(inputs)
        y = self._position_embedding(inputs)
        z = self._add([x, y])
        if self._dropout is not None:
            z = self._dropout(z, training=training)
        return z

    def compute_mask(self, inputs, mask=None):
        return self._token_embedding.compute_mask(inputs=inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        return self._token_embedding.compute_output_shape(input_shape=input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocabulary_size': self.vocabulary_size,
            'sequence_length': self.sequence_length,
            'embedding_dimension': self.embedding_dimension,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'mask_zero': self.mask_zero,
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
                 padding='same',
                 data_format=None,
                 convolution_groups=1,
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
        strides = size if strides is None else strides

        self.size = size
        self.embedding_dimension = embedding_dimension
        self.mode = mode
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.convolution_groups = convolution_groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.supports_masking = True

        if mode == 'patch':
            self._patch_extractor = PatchExtractor2D(size=size, strides=strides, padding=padding)
            self._dense = keras.layers.Dense(
                units=embedding_dimension, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        elif mode == 'convolution':
            self._convolution = keras.layers.Convolution2D(
                filters=embedding_dimension, kernel_size=size, strides=strides, padding=padding,
                data_format=data_format, groups=convolution_groups, activation=None, use_bias=use_bias,
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint)
        else:
            raise ValueError(f'Unexpected value for `mode`: {mode}. Supported values are: `convolution` or `patch`.')

    def call(self, inputs, **kwargs):
        if self.mode == 'patch':
            return self._dense(self._patch_extractor(inputs))
        else:
            return self._convolution(inputs)

    def compute_output_shape(self, input_shape):
        if self.mode == 'patch':
            output_shape = self._patch_extractor.compute_output_shape(input_shape=input_shape)
            return self._dense.compute_output_shape(input_shape=output_shape)
        else:
            return self._convolution.compute_output_shape(input_shape=input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size,
            'embedding_dimension': self.embedding_dimension,
            'mode': self.mode,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'convolution_groups': self.convolution_groups,
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


@keras.saving.register_keras_serializable(package='tfe.layers')
class PatchAndPositionEmbedding2D(keras.layers.Layer):
    def __init__(self,
                 size,
                 embedding_dimension,
                 mode='convolution',
                 strides=None,
                 padding='same',
                 data_format=None,
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
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.size = size
        self.embedding_dimension = embedding_dimension
        self.mode = mode
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
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
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self._patch_embedding = PatchEmbedding2D(
            size=size, embedding_dimension=embedding_dimension, mode=mode, strides=strides, padding=padding,
            data_format=data_format, convolution_groups=convolution_groups, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self._position_embedding = PositionEmbedding2D(
            size=size, embedding_dimension=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, embeddings_constraint=embeddings_constraint)
        self._add = keras.layers.Add()
        self._dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        x = self._patch_embedding(inputs)
        y = self._position_embedding(inputs)
        # TODO: Revise this
        z = self._add([ops.reshape(x, newshape=[-1, x.shape[1] * x.shape[2], x.shape[3]]), y])
        if self._dropout is not None:
            z = self._dropout(z, training=training)
        return z

    def compute_output_shape(self, input_shape):
        output_shape = self._patch_embedding.compute_output_shape(input_shape=input_shape)
        return self._position_embedding.compute_output_shape(input_shape=output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size,
            'embedding_dimension': self.embedding_dimension,
            'mode': self.mode,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
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
            'embeddings_constraint': self.embeddings_constraint,
            'rate': self.rate,
            'seed': self.seed
        })
        return config
