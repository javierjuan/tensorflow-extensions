import tensorflow as tf


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 add_batch_size_dimension=False,
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
        self.add_batch_size_dimension = add_batch_size_dimension

        self.embedding = self.add_weight(
            shape=(1, input_dim, output_dim) if add_batch_size_dimension else (input_dim, output_dim),
            initializer=embeddings_initializer, name='fixed_embedding', regularizer=embeddings_regularizer,
            constraint=embeddings_constraint, experimental_autocast=False)

    def call(self, batch_size=None, **kwargs):
        if self.add_batch_size_dimension and batch_size is not None:
            return tf.tile(self.embedding, multiples=[batch_size, 1, 1])
        else:
            return self.embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'add_batch_size_dimension': self.add_batch_size_dimension,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint
        })
        return config
