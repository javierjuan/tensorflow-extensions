import tensorflow as tf


class Dice(tf.keras.metrics.FBetaScore):
    def __init__(self, average=None, beta=1.0, threshold=None, name=None):
        super().__init__(average=average, beta=beta, threshold=threshold, name=name)
