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
        self.posterior = tf.keras.layers.Dense(units=num_classes, activation='softmax')

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
        return self.posterior(x)
