import tensorflow as tf

from ..matchers import HungarianMatcher
from ..matchers.hungarian import compute_ciou


def compute_label_loss(y_true, y_pred, assignment, padding_mask, padding_weight=0.1):
    y_true_indices, y_pred_indices = tf.unstack(tf.where(assignment), num=2, axis=-1)
    loss = tf.keras.losses.categorical_crossentropy(y_true=tf.gather(y_true, y_true_indices),
                                                    y_pred=tf.gather(y_pred, y_pred_indices),
                                                    from_logits=False, label_smoothing=0)
    loss *= ((1.0 - padding_mask) + (padding_mask * padding_weight))
    return tf.math.reduce_sum(loss)


def compute_bounding_box_loss(y_true, y_pred, assignment, padding_mask):
    y_true_indices, y_pred_indices = tf.unstack(tf.where(assignment), num=2, axis=-1)
    scores = compute_ciou(boxes1=tf.gather(y_true, y_true_indices), boxes2=tf.gather(y_pred, y_pred_indices))
    loss = 1.0 - tf.linalg.tensor_diag_part(scores)
    loss *= (1.0 - padding_mask)
    return tf.math.reduce_sum(loss)


class DETR(tf.keras.losses.Loss):
    def __init__(self, labels_weight=1.0, bounding_boxes_weight=1.0, padding_axis=-1,
                 reduction=tf.keras.losses.Reduction.AUTO, name='detr'):
        super().__init__(name=name, reduction=reduction)
        self.matcher = HungarianMatcher(label_weight=labels_weight, bounding_box_weight=bounding_boxes_weight,
                                        padding_axis=padding_axis)
        self.padding_axis = padding_axis
        self.label_weight = labels_weight
        self.bounding_box_weight = bounding_boxes_weight

    def call(self, y_true, y_pred):
        zip_args = (tf.unstack(y_true['label']), tf.unstack(y_true['bounding_box']),
                    tf.unstack(y_pred['label']), tf.unstack(y_pred['bounding_box']))
        batch_loss = []
        for y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box in zip(*zip_args):
            # Compute Hungarian matching
            assignment = self.matcher(y_true_label=y_true_label, y_true_bounding_box=y_true_bounding_box,
                                      y_pred_label=y_pred_label, y_pred_bounding_box=y_pred_bounding_box)
            # Compute loss based on Hungarian matched pairs
            padding_mask = y_true_label[..., self.padding_axis]
            label_loss = compute_label_loss(y_true=y_true_label, y_pred=y_pred_label, assignment=assignment,
                                            padding_mask=padding_mask, padding_weight=0.1)
            bounding_box_loss = compute_bounding_box_loss(y_true=y_true_bounding_box, y_pred=y_pred_bounding_box,
                                                          assignment=assignment, padding_mask=padding_mask)
            loss = label_loss + bounding_box_loss
            batch_loss.append(loss)
        return tf.convert_to_tensor(batch_loss, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_weight': self.label_weight,
            'bounding_box_weight': self.bounding_box_weight,
        })
        return config
