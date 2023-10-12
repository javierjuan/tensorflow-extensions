import tensorflow as tf


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 sequence_length,
                 vocabulary_size,
                 embedding_dimension,
                 positional='embedding',
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
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.positional = positional
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

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dimension, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length,
            sparse=sparse)
        if positional == 'encoding':
            self.positional_encoding = PositionalEncoding(max_wavelength=max_wavelength)
        elif positional == 'embedding':
            self.positional_encoding = PositionalEmbedding(
                sequence_length=sequence_length, embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint, input_length=input_length, sparse=sparse)
        else:
            raise ValueError(f'Unexpected `positional` value: {positional}. Possible values are `encoding` or '
                             '`embedding`')
        self.dropout = tf.keras.layers.Dropout(rate=rate, seed=seed) if rate is not None else None

    def call(self, inputs, training=False, **kwargs):
        x = self.token_embedding(inputs)
        x = self.positional_encoding(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'vocabulary_size': self.vocabulary_size,
            'embedding_dimension': self.embedding_dimension,
            'positional': self.positional,
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


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,
                 max_wavelength=10000,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_wavelength = max_wavelength
        self.supports_masking = True

        self.positions_mask = None
        self.timescales = None
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        embedding_dimension = input_shape[-1]
        timescales = tf.math.pow(tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype),
                                 (tf.cast(2 * (tf.range(embedding_dimension) // 2), self.compute_dtype) /
                                  tf.cast(embedding_dimension, self.compute_dtype)))
        self.timescales = tf.expand_dims(timescales, axis=0)
        self.positions_mask = tf.cast(tf.range(embedding_dimension) % 2, self.compute_dtype)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        positions = tf.cast(tf.range(start=0, limit=input_shape[-2], delta=1), self.compute_dtype)
        angles = tf.expand_dims(positions, axis=1) * self.timescales
        encoding = (tf.math.sin(angles) * (1 - self.positions_mask) + tf.math.cos(angles) * self.positions_mask)
        return self.add([inputs, tf.broadcast_to(encoding, shape=input_shape)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_wavelength': self.max_wavelength
        })


class PositionalEmbedding(tf.keras.layers.Layer):
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
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        embedding_dimension = input_shape[-1]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.sequence_length, output_dim=embedding_dimension,
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            activity_regularizer=self.activity_regularizer, embeddings_constraint=self.embeddings_constraint,
            mask_zero=False, input_length=self.input_length, sparse=self.sparse)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        positions = tf.cast(tf.range(start=0, limit=input_shape[-2], delta=1), self.compute_dtype)
        return self.add([inputs, tf.broadcast_to(self.embedding(positions), shape=input_shape)])

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


class PositionalEmbedding2D(tf.keras.layers.Layer):
    def __init__(self,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 input_length=None,
                 sparse=False,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.activity_regularizer = activity_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.input_length = input_length
        self.sparse = sparse
        self.supports_masking = True

        self.row_embedding = None
        self.col_embedding = None
        self.concatenate = tf.keras.layers.Concatenate()
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        width = input_shape[-2]
        height = input_shape[-3]
        embedding_dimension = input_shape[-1] // 2
        self.row_embedding = tf.keras.layers.Embedding(
            input_dim=width, output_dim=embedding_dimension,
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            activity_regularizer=self.activity_regularizer, embeddings_constraint=self.embeddings_constraint,
            mask_zero=False, input_length=self.input_length, sparse=self.sparse)
        self.col_embedding = tf.keras.layers.Embedding(
            input_dim=height, output_dim=embedding_dimension,
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            activity_regularizer=self.activity_regularizer, embeddings_constraint=self.embeddings_constraint,
            mask_zero=False, input_length=self.input_length, sparse=self.sparse)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        row_positions = tf.cast(tf.range(start=0, limit=input_shape[-2], delta=1), self.compute_dtype)
        col_positions = tf.cast(tf.range(start=0, limit=input_shape[-3], delta=1), self.compute_dtype)
        embedding = self.concatenate([self.row_embedding(row_positions), self.col_embedding(col_positions)])
        return self.add([inputs, tf.broadcast_to(embedding, shape=input_shape)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'input_length': self.input_length,
            'sparse': self.sparse
        })


class PatchEncoding2D(tf.keras.layers.Layer):
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
                 mask_zero=False,
                 input_length=None,
                 sparse=False,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        strides = size if strides is None else strides
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
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.sparse = sparse
        self.supports_masking = True

        self.class_token = self.add_weight(shape=[1, 1, embedding_dimension], name='class_token', trainable=True)
        if mode in ('patch', 'crop'):
            self.dense = tf.keras.layers.Dense(
                units=embedding_dimension, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        elif mode in ('convolution', 'conv'):
            self.convolution = tf.keras.layers.Convolution2D(
                filters=embedding_dimension, kernel_size=size, strides=strides, padding=padding,
                data_format=data_format, dilation_rate=dilation_rate, groups=convolution_groups, activation=None,
                use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint)
        else:
            raise ValueError(f'Unexpected value for `mode`: {mode}. Possible values are: `convolution`, `conv`, '
                             f'`patch` or `crop`.')
        self.add = tf.keras.layers.Add()
        self.embedding = None
        self.positions = None

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_shape[-2], output_dim=self.embedding_dimension,
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            activity_regularizer=self.activity_regularizer, embeddings_constraint=self.embeddings_constraint,
            mask_zero=self.mask_zero, input_length=self.input_length, sparse=self.sparse)
        self.positions = tf.expand_dims(tf.range(start=0, limit=input_shape[1], delta=1), axis=0)

    def call(self, inputs, **kwargs):
        # TODO: Class Token
        if self.mode in ('patch', 'crop'):
            patches = tf.image.extract_patches(images=inputs, sizes=[1, *self.size, 1], strides=[1, *self.strides, 1],
                                               rates=[1, *self.rates, 1], padding=self.padding.upper())
            patches = self.dense(patches)
        else:
            patches = self.convolution(inputs)
        patches = tf.reshape(patches, shape=[-1, patches.shape[1] * patches.shape[2], patches.shape[3]])
        return self.add([patches, self.embedding(self.positions)])

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
            'embeddings_constraint': self.embeddings_constraint,
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
            'sparse': self.sparse
        })
