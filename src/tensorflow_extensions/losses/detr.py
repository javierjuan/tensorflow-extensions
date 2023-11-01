import tensorflow as tf
from keras_cv.bounding_box import compute_ciou

from ..ops import linear_sum_assignment


class DETR(tf.keras.losses.Loss):
    def __init__(self, label_weight=1.0, bounding_box_weight=1.0,
                 reduction=tf.keras.losses.Reduction.AUTO, name='detr'):
        super().__init__(name=name, reduction=reduction)
        self.label_weight = label_weight
        self.bounding_box_weight = bounding_box_weight

    def call(self, y_true, y_pred):
        zip_args = (tf.unstack(y_true['label']), tf.unstack(y_true['bounding_box']),
                    tf.unstack(y_pred['label']), tf.unstack(y_pred['bounding_box']))
        batch_loss = []
        for y_true_labels, y_true_bounding_box, y_pred_labels, y_pred_bounding_box in zip(*zip_args):
            non_padding_mask = tf.cast(1.0 - y_true_labels[..., -1], dtype=tf.bool)
            if tf.math.reduce_any(non_padding_mask):
                # Compute cost for Hungarian matching
                label_scores = 1.0 - tf.matmul(y_true_labels, y_pred_labels, transpose_b=True)
                bounding_box_scores = 1.0 - compute_ciou(boxes1=y_true_bounding_box, boxes2=y_pred_bounding_box,
                                                         bounding_box_format='rel_xyxy')
                cost_matrix = self.label_weight * label_scores + self.bounding_box_weight * bounding_box_scores
                cost_matrix = tf.boolean_mask(cost_matrix, non_padding_mask, axis=0)
                assignment = tf.stop_gradient(linear_sum_assignment(cost_matrix))
                # Compute loss based on Hungarian matched pairs and unmatched pairs
                assigned_loss = tf.expand_dims(tf.gather_nd(cost_matrix, tf.where(assignment)), axis=-1)
                unassigned_loss = tf.gather(y_pred_labels[..., -1], tf.where(tf.reduce_all(~assignment, axis=0)))
                loss = tf.reduce_sum(tf.concat([assigned_loss, 0.1 * -tf.math.log(unassigned_loss)], axis=0))
            else:
                loss = tf.math.reduce_sum(0.1 * -tf.math.log(y_pred_labels[..., -1]))
            batch_loss.append(loss)
        return tf.convert_to_tensor(batch_loss, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_weight': self.label_weight,
            'bounding_box_weight': self.bounding_box_weight,
        })
        return config
