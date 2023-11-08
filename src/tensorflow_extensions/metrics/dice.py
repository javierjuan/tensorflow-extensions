import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package='tfe.metrics')
class Dice(tf.keras.metrics.FBetaScore):
    def __init__(self, average=None, beta=1.0, threshold=None, name=None):
        super().__init__(average=average, beta=beta, threshold=threshold, name=name)
