import tensorflow as tf


class Jaccard(tf.keras.metrics.FBetaScore):
    def __init__(self, average=None, beta=1.0, threshold=None, name=None):
        super().__init__(average=average, beta=beta, threshold=threshold, name=name)

    def result(self):
        score = super().result()
        return tf.math.divide_no_nan(score, (2 - score))
