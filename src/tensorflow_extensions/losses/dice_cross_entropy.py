import keras_core as keras
import tensorflow as tf
from keras_core import ops

from .dice import dice_score
from .utils import initialize_loss, finalize_loss


@tf.function
@keras.saving.register_keras_serializable(package='tfe.losses')
def _cross_entropy_loss(y_true, y_pred):
    loss = keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False, label_smoothing=0)
    # Reduce to be compatible with the other losses
    axis_reduce = ops.arange(start=1, stop=ops.ndim(loss))
    return ops.mean(loss, axis=axis_reduce)


@keras.saving.register_keras_serializable(package='tfe.losses')
class DicePlusCategoricalCrossEntropy(keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, label_penalties=None, from_logits=False, reduction='sum_over_batch_size',
                 name='dice_plus_categorical_cross_entropy'):
        super().__init__(name=name, reduction=reduction)

        self.label_smoothing = label_smoothing
        self.label_penalties = label_penalties
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true, y_pred = initialize_loss(y_true=y_true, y_pred=y_pred, label_smoothing=self.label_smoothing,
                                         from_logits=self.from_logits)
        loss_cross_entropy = _cross_entropy_loss(y_true=y_true, y_pred=y_pred)
        loss_dice = 1.0 - dice_score(y_true=y_true, y_pred=y_pred)
        return finalize_loss(loss=loss_cross_entropy + loss_dice, label_penalties=self.label_penalties)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'label_penalties': self.label_penalties,
            'from_logits': self.from_logits
        })
        return config
