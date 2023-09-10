import tensorflow as tf
from src import layers, augmentation, preprocessing


class Model(tf.keras.Model):
    def train_step(self, data):
        x, y, sample_weight = data if len(data) == 3 else data + (None,)

        with tf.GradientTape() as tape:
            with tf.name_scope('Forward'):
                y_pred = self(x, training=True)
                with tf.name_scope('Loss'):
                    loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        with tf.name_scope('Backward'):
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        with tf.name_scope('Metrics'):
            self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
            return {metric.name: metric.result() for metric in self.metrics}


class Transformer(Model):
    def __init__(self,
                 input_vocabulary_size,
                 output_vocabulary_size,
                 embedding_dimension,
                 dense_dimension,
                 num_heads,
                 num_encoders=1,
                 num_decoders=1,
                 positional='embedding',
                 max_input_length=1024,
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
            max_input_length=max_input_length, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length, sparse=sparse,
            rate=rate, seed=seed)
        self.output_embedding = layers.TokenAndPositionEmbedding(
            vocabulary_size=output_vocabulary_size, embedding_dimension=embedding_dimension, positional=positional,
            max_input_length=max_input_length, embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer, activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint, mask_zero=mask_zero, input_length=input_length, sparse=sparse,
            rate=rate, seed=seed)
        self.encoders = [layers.TransformerEncoder(
            embedding_dimension=embedding_dimension, dense_dimension=dense_dimension, num_heads=num_heads,
            use_bias=use_bias, output_shape=output_shape, attention_axes=attention_axes,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed) for _ in range(num_encoders)]
        self.decoders = [layers.TransformerDecoder(
            embedding_dimension=embedding_dimension, dense_dimension=dense_dimension, num_heads=num_heads,
            use_bias=use_bias, output_shape=output_shape, attention_axes=attention_axes,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
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
            bias_constraint=bias_constraint)

    def call(self, inputs, training=False, return_attention_scores=False, **kwargs):
        inputs, outputs = inputs
        inputs = self.input_embedding(inputs, training=training)
        for encoder in self.encoders:
            inputs = encoder(inputs, training=training, return_attention_scores=return_attention_scores)
        outputs = self.output_embedding(outputs, training=training)
        for decoder in self.decoders:
            outputs = decoder(outputs, context=inputs, training=training,
                              return_attention_scores=return_attention_scores)
        return self.posteriors(outputs)


class ViT(Model):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_transformer_encoder_blocks,
                 embedding_dimension,
                 dense_dimension,
                 num_heads,
                 units,
                 num_classes,
                 epsilon=1e-6,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.augmentation = augmentation.ImageRGB()
        self.preprocessing = preprocessing.Standard2D(height=image_size[0], width=image_size[1])
        self.patch_encoder = layers.PatchEncoding2D(size=patch_size, embedding_dimension=embedding_dimension,
                                                    mode='convolution')
        self.transformer_encoder_blocks = [layers.TransformerEncoder(embedding_dimension=embedding_dimension,
                                                                     dense_dimension=dense_dimension,
                                                                     num_heads=num_heads)
                                           for _ in range(num_transformer_encoder_blocks)]
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_block_1 = layers.DenseBlock(units=units, normalization='layer', epsilon=epsilon)
        self.dense_block_2 = layers.DenseBlock(units=units // 4, normalization='layer', epsilon=epsilon)
        self.posteriors = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def adapt(self, data, batch_size=None, steps=None):
        self.preprocessing.adapt(data=data, batch_size=batch_size, steps=steps)

    def call(self, inputs, training=False, **kwargs):
        x = self.augmentation(inputs, training=training)
        x = self.preprocessing(x)
        x = self.patch_encoder(x)
        for transformer_encoder_block in self.transformer_encoder_blocks:
            x = transformer_encoder_block(x, training=training)
        x = self.normalization(x)
        x = self.flatten(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        return self.posteriors(x)
