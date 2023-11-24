import keras_core as keras


@keras.saving.register_keras_serializable(package='tfe.metrics')
class Dice(keras.metrics.FBetaScore):
    def __init__(self, average=None, threshold=None, name=None):
        super().__init__(average=average, beta=1.0, threshold=threshold, name=name)
