import keras
from keras import ops

from .embedding import FixedEmbedding
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

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        positions = ops.cast(ops.arange(inputs.shape[-2]), dtype=self.compute_dtype)
        angles = ops.expand_dims(positions, axis=1) * self.timescales
        encoding = ops.sin(angles) * (1 - self.positions_mask) + ops.cos(angles) * self.positions_mask
        return self.add([inputs, ops.broadcast_to(encoding, shape=inputs.shape)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_wavelength': self.max_wavelength
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class PositionalEmbedding1D(keras.layers.Layer):
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

        self.embedding = None
        self.add = keras.layers.Add()

    def build(self, input_shape):
        # TODO: This is possibly incorrect because `input_dim` cannot be extrapolated only from the first sample used in
        #  the training process (which is the one used to run the `build()` method. If so, we are assuming that there
        #  are not ragged tensors, and hence we can replace this `Embedding` with a `FixedEmbedding` layer
        self.embedding = keras.layers.Embedding(
            input_dim=input_shape[-2], output_dim=input_shape[-1],
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            embeddings_constraint=self.embeddings_constraint, mask_zero=False)
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        positions = ops.cast(ops.arange(inputs.shape[-2]), dtype=self.compute_dtype)
        return self.add([inputs, ops.broadcast_to(self.embedding(positions), shape=inputs.shape)])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint
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

        self.embedding = None
        self.add = keras.layers.Add()

    def build(self, input_shape):
        self.embedding = FixedEmbedding(
            input_dim=input_shape[-2] * input_shape[-3], output_dim=input_shape[-1],
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            embeddings_constraint=self.embeddings_constraint)
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        embedding = self.embedding(batch_size=inputs.shape[0])
        inputs = ops.reshape(inputs, new_shape=[-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]])
        return self.add([inputs, embedding])

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] * input_shape[2], input_shape[3]

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
        self.supports_masking = True

        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero)
        self.positional_encoding = PositionalEncoding1D(max_wavelength=max_wavelength)
        self.dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        x = self.token_embedding(inputs)
        x = self.positional_encoding(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = self.token_embedding.compute_output_shape(input_shape=input_shape)
        return self.positional_encoding.compute_output_shape(input_shape=output_shape)

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
        self.embedding_dimension = embedding_dimension
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.mask_zero = mask_zero
        self.rate = rate
        self.seed = seed
        self.supports_masking = True

        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero)
        self.positional_embedding = PositionalEmbedding1D(
            embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint)
        self.dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        x = self.token_embedding(inputs)
        x = self.positional_embedding(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = self.token_embedding.compute_output_shape(input_shape=input_shape)
        return self.positional_embedding.compute_output_shape(input_shape=output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocabulary_size': self.vocabulary_size,
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
            self.patch_extractor = PatchExtractor2D(size=size, strides=strides, padding=padding)
            self.dense = keras.layers.Dense(
                units=embedding_dimension, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        elif mode == 'convolution':
            self.convolution = keras.layers.Convolution2D(
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
            patches = self.patch_extractor(inputs)
            return self.dense(patches)
        else:
            return self.convolution(inputs)

    def compute_output_shape(self, input_shape):
        if self.mode == 'patch':
            output_shape = self.patch_extractor.compute_output_shape(input_shape=input_shape)
            return self.dense.compute_output_shape(input_shape=output_shape)
        else:
            return self.convolution.compute_output_shape(input_shape=input_shape)

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

        self.patch_embedding = PatchEmbedding2D(
            size=size, embedding_dimension=embedding_dimension, mode=mode, strides=strides, padding=padding,
            data_format=data_format, convolution_groups=convolution_groups, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.positional_embedding = PositionalEmbedding2D(
            embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint)
        self.dropout = keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        x = self.patch_embedding(inputs)
        x = self.positional_embedding(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = self.patch_embedding.compute_output_shape(input_shape=input_shape)
        return self.positional_embedding.compute_output_shape(input_shape=output_shape)

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
