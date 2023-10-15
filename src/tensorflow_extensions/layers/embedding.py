import tensorflow as tf


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 input_dimensions,
                 output_dimensions,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.embedding = None

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(self.input_dimensions, self.output_dimensions), initializer=self.embeddings_initializer,
            name='fixed_embedding', regularizer=self.embeddings_regularizer, constraint=self.embeddings_constraint,
            experimental_autocast=False)
        super().build(input_shape=input_shape)

    def call(self, inputs=None, **kwargs):
        return tf.expand_dims(self.embedding, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dimensions': self.input_dimensions,
            'output_dimensions': self.output_dimensions,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint
        })
        return config
