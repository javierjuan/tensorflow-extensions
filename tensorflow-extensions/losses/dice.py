import keras
from keras import ops
import tensorflow as tf

from .utils import initialize_loss, finalize_loss


@tf.function
@keras.saving.register_keras_serializable(package='tfe.losses')
def dice_score(y_true, y_pred):
    # IMPORTANT: The denominator MUST be squared for mathematical correctness. Dice metric is defined for discrete
    # sets of labels. For continuous probability arrays the denominator must be squared in order to accomplish with
    # the concept of `cardinality` -> |A| = sum(a_i^2).
    axis_reduce = ops.arange(start=1, limit=ops.ndim(y_pred) - 1)
    numerator = 2.0 * ops.sum(y_true * y_pred, axis=axis_reduce)
    denominator = ops.sum(ops.square(y_true) + ops.square(y_pred), axis=axis_reduce)
    return ops.where(ops.equal(denominator, 0.0), ops.zeros_like(numerator), numerator / denominator)


@keras.saving.register_keras_serializable(package='tfe.losses')
class Dice(keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, label_penalties=None, from_logits=False, reduction='sum_over_batch_size', name='dice'):
        super().__init__(name=name, reduction=reduction)
        self.label_smoothing = label_smoothing
        self.label_penalties = label_penalties
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true, y_pred = initialize_loss(y_true=y_true, y_pred=y_pred, label_smoothing=self.label_smoothing,
                                         from_logits=self.from_logits)
        loss = 1.0 - dice_score(y_true=y_true, y_pred=y_pred)
        return finalize_loss(loss=loss, label_penalties=self.label_penalties)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'label_penalties': self.label_penalties,
            'from_logits': self.from_logits
        })
        return config
