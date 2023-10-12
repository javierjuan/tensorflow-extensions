import tensorflow as tf

from .utils import initialize_loss, finalize_loss


@tf.function
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

    axis_reduce = tf.range(start=1, limit=tf.rank(y_pred) - 1)
    numerator = tf.math.reduce_sum(tp, axis=axis_reduce)
    denominator = tf.math.reduce_sum(tp + tf.math.square(fp) + tf.math.square(fn), axis=axis_reduce)
    return tf.math.divide_no_nan(numerator, denominator)


class Tversky(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.5, label_smoothing=0.1, label_penalties=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='tversky'):
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
