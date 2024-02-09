import keras
import tensorflow as tf
from keras import ops

from .utils import initialize_loss, finalize_loss


@tf.function
@keras.saving.register_keras_serializable(package='tfe.losses')
def tversky_score(y_true, y_pred, alpha=0.5, beta=0.5):
    # WARNING!: This is not a mathematically correct formula. p1 = 1 - y_pred and g1 = 1 - y_true try to imitate the
    # operation `relative complement` (|A\B|) of two discrete sets of labels, but p1 and g1 are not equivalents of
    # this operation. Therefore, the metric does not behave correctly for perfectly matching target and labels

    p0 = y_pred
    p1 = 1 - y_pred
    g0 = y_true
    g1 = 1 - y_true

    tp = p0 * g0
    fp = alpha * p0 * g1
    fn = beta * p1 * g0

    axis_reduce = ops.arange(start=1, stop=ops.ndim(y_pred) - 1)
    numerator = ops.sum(tp, axis=axis_reduce)
    denominator = ops.sum(tp + ops.square(fp) + ops.square(fn), axis=axis_reduce)
    return ops.where(ops.equal(denominator, 0.0), ops.zeros_like(numerator), numerator / denominator)


@keras.saving.register_keras_serializable(package='tfe.losses')
class Tversky(keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.5, label_smoothing=0.1, label_penalties=None, from_logits=False,
                 reduction='sum_over_batch_size', name='tversky'):
        super().__init__(name=name, reduction=reduction)

        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.label_penalties = label_penalties
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true, y_pred = initialize_loss(y_true=y_true, y_pred=y_pred, label_smoothing=self.label_smoothing,
                                         from_logits=self.from_logits)
        loss = 1.0 - tversky_score(y_true=y_true, y_pred=y_pred, alpha=self.alpha, beta=self.beta)
        return finalize_loss(loss=loss, label_penalties=self.label_penalties)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'label_smoothing': self.label_smoothing,
            'label_penalties': self.label_penalties,
            'from_logits': self.from_logits
        })
        return config
