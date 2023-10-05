import tensorflow as tf
from src import layers, augmentation, preprocessing


class Model(tf.keras.Model):
    def train_step(self, data):
        x, y, sample_weight = data if len(data) == 3 else data + (None,)

        with tf.GradientTape() as tape:
            with tf.name_scope(name='Forward'):
                y_pred = self(x, training=True)
                with tf.name_scope('Loss'):
                    loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        with tf.name_scope(name='Backward'):
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        with tf.name_scope(name='Metrics'):
            self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
            return {metric.name: metric.result() for metric in self.metrics}


class Seq2SeqTransformer(Model):
    def __init__(self,
                 input_vocabulary_size,
                 output_vocabulary_size,
                 sequence_length,
                 embedding_dimension,
                 units,
                 num_heads,
                 num_encoders=1,
                 num_decoders=1,
                 positional='embedding',
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
        self.input_embedding = layers.TokenAndPositionEmbedding(
            vocabulary_size=input_vocabulary_size, embedding_dimension=embedding_dimension, positional=positional,
            sequence_length=sequence_length, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length, sparse=sparse,
            rate=rate, seed=seed)
        self.output_embedding = layers.TokenAndPositionEmbedding(
            vocabulary_size=output_vocabulary_size, embedding_dimension=embedding_dimension, positional=positional,
            sequence_length=sequence_length, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length, sparse=sparse,
            rate=rate, seed=seed)
        self.encoders = [layers.TransformerEncoder(
            units=units, num_heads=num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed) for _ in range(num_encoders)]
        self.decoders = [layers.TransformerDecoder(
            units=units, num_heads=num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=activity_regularizer, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed) for _ in range(num_decoders)]
        self.posteriors = tf.keras.layers.Dense(
            units=output_vocabulary_size, activation='softmax', use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, dtype=tf.float32)

    def call(self, inputs, training=False, **kwargs):
        inputs, outputs = inputs
        inputs = self.input_embedding(inputs, training=training)
        for encoder in self.encoders:
            inputs = encoder(inputs, training=training)
        outputs = self.output_embedding(outputs, training=training)
        for decoder in self.decoders:
            outputs = decoder(outputs, context=inputs, training=training)
        return self.posteriors(outputs)


class ViT(Model):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 embedding_dimension,
                 encoder_units,
                 dense_units,
                 num_heads,
                 num_encoders=1,
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
                 flip_mode='horizontal_and_vertical',
                 rotation_factor=0.5,
                 zoom_factor=0.5,
                 brightness_factor=0.5,
                 contrast_factor=0.5,
                 value_range=(0, 255),
                 fill_mode='reflect',
                 interpolation='bilinear',
                 fill_value=0.0,
                 crop_to_aspect_ratio=False,
                 mean=None,
                 variance=None,
                 invert=False,
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
        self.augmentation = augmentation.ImageRGB(
            flip_mode=flip_mode, rotation_factor=rotation_factor, zoom_factor=zoom_factor,
            brightness_factor=brightness_factor, contrast_factor=contrast_factor, value_range=value_range,
            fill_mode=fill_mode, interpolation=interpolation, seed=seed, fill_value=fill_value)
        self.preprocessing = preprocessing.Standard2D(
            height=image_size[0], width=image_size[1], crop_to_aspect_ratio=crop_to_aspect_ratio, mean=mean,
            variance=variance, invert=invert, interpolation=interpolation, axis=axis)
        self.patch_encoding = layers.PatchEncoding2D(
            size=patch_size, embedding_dimension=embedding_dimension, mode=mode, strides=strides, rates=rates,
            padding=padding, data_format=data_format, dilation_rate=dilation_rate,
            convolution_groups=convolution_groups, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero, input_length=input_length, sparse=sparse)
        self.encoders = [layers.TransformerEncoder(
            units=encoder_units, num_heads=num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed) for _ in range(num_encoders)]
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_block_1 = layers.DenseBlock(
            units=dense_units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            synchronized=synchronized, axis=axis, rate=rate, seed=seed)
        self.dense_block_2 = layers.DenseBlock(
            units=dense_units // 4, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            synchronized=synchronized, axis=axis, rate=rate, seed=seed)
        self.posteriors = tf.keras.layers.Dense(
            units=num_classes, activation='softmax', use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, dtype=tf.float32)

    def adapt(self, data, batch_size=None, steps=None):
        self.preprocessing.adapt(data=data, batch_size=batch_size, steps=steps)

    def call(self, inputs, training=False, **kwargs):
        x = self.augmentation(inputs, training=training)
        x = self.preprocessing(x)
        x = self.patch_encoding(x)
        for encoder in self.encoders:
            x = encoder(x, training=training)
        x = self.normalization(x)
        x = self.flatten(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        return self.posteriors(x)
