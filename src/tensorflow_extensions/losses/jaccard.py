import tensorflow as tf

from .dice import dice_score
from .utils import initialize_loss, finalize_loss


class Jaccard(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, label_penalties=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='jaccard'):
        super().__init__(name=name, reduction=reduction)

        self.label_smoothing = label_smoothing
        self.label_penalties = label_penalties
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true, y_pred = initialize_loss(y_true=y_true, y_pred=y_pred, label_smoothing=self.label_smoothing,
                                         from_logits=self.from_logits)
        scores = dice_score(y_true=y_true, y_pred=y_pred)
        loss = 1.0 - (scores / (2.0 - scores))
        return finalize_loss(loss=loss, label_penalties=self.label_penalties)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'label_penalties': self.label_penalties,
            'from_logits': self.from_logits
        })
        return config
