import tensorflow as tf
from src import layers


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
                 patch_size,
                 num_classes,
                 embedding_dimension,
                 encoder_units,
                 dense_units,
                 num_heads,
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
            units=_encoder_units, num_heads=num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed) for _encoder_units in range(encoder_units)]
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_blocks = [layers.DenseBlock(
            units=_dense_units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, normalization=normalization, momentum=momentum, epsilon=epsilon,
            normalization_groups=normalization_groups, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            synchronized=synchronized, axis=axis, rate=rate, seed=seed) for _dense_units in dense_units]
        self.posteriors = tf.keras.layers.Dense(
            units=num_classes, activation='softmax', use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, dtype=tf.float32)

    def call(self, inputs, training=False, **kwargs):
        x = self.patch_encoding(inputs)
        for encoder in self.encoders:
            x = encoder(x, training=training)
        x = self.normalization(x)
        x = self.flatten(x)
        for dense_block in self.dense_blocks:
            x = dense_block(x, training=training)
        return self.posteriors(x)
