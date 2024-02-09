import keras
from keras import ops


@keras.saving.register_keras_serializable(package='tfe.metrics')
class Jaccard(keras.metrics.FBetaScore):
    def __init__(self, average=None, threshold=None, name=None):
        super().__init__(average=average, beta=1.0, threshold=threshold, name=name)

    def result(self):
        score = super().result()
        return score / (2 - score)
