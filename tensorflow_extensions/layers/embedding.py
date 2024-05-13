import keras
from keras import ops


@keras.saving.register_keras_serializable(package='tfe.layers')
class FixedEmbedding(keras.layers.Layer):
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

        self._embedding = self.add_weight(
            shape=(input_dim, output_dim), initializer=embeddings_initializer, regularizer=embeddings_regularizer,
            constraint=embeddings_constraint, trainable=True, name='fixed_embedding')

    def call(self, inputs, **kwargs):
        return ops.tile(self._embedding, repeats=(inputs.shape[0], 1, 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.input_dim, self.output_dim

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
