import tensorflow as tf

from .utils import initialize_loss, finalize_loss


@tf.function
@tf.keras.saving.register_keras_serializable(package='tfe.losses')
def dice_score(y_true, y_pred):
    # IMPORTANT: The denominator MUST be squared for mathematical correctness. Dice metric is defined for discrete
    # sets of labels. For continuous probability arrays the denominator must be squared in order to accomplish with
    # the concept of `cardinality` -> |A| = sum(a_i^2).
    axis_reduce = tf.range(start=1, limit=tf.rank(y_pred) - 1)
    numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis_reduce)
    denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=axis_reduce)
    return tf.math.divide_no_nan(numerator, denominator)


@tf.keras.saving.register_keras_serializable(package='tfe.losses')
class Dice(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, label_penalties=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='dice'):
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
