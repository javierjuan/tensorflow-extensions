import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

        self.embedding = self.add_weight(
            shape=(input_dim, output_dim), initializer=embeddings_initializer, regularizer=embeddings_regularizer,
            constraint=embeddings_constraint, name='fixed_embedding')

    def call(self, batch_size=None, **kwargs):
        if batch_size is None:
            return self.embedding
        else:
            x = tf.expand_dims(self.embedding, axis=0)
            multiples = tf.one_hot(indices=0, depth=tf.rank(x), on_value=batch_size, off_value=1, dtype=tf.int32)
            return tf.tile(x, multiples=multiples)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint
        })
        return config
