import numpy as np
import tensorflow as tf
from keras_cv.bounding_box import compute_iou
from scipy.optimize import linear_sum_assignment


@tf.numpy_function(Tout=tf.bool)
def tf_linear_sum_assignment(cost_matrix):
    rows, cols = linear_sum_assignment(cost_matrix)
    assignment = np.zeros_like(cost_matrix, dtype=bool)
    assignment[rows, cols] = True
    return assignment


class DETR(tf.keras.losses.Loss):
    def __init__(self, label_weight=1.0, bounding_box_weight=1.0, iou_weight=1.0,
                 reduction=tf.keras.losses.Reduction.AUTO, name='detr'):
        super().__init__(name=name, reduction=reduction)
        self.label_weight = label_weight
        self.bounding_box_weight = bounding_box_weight
        self.iou_weight = iou_weight

    def call(self, y_true, y_pred):
        zip_args = (tf.unstack(y_true['label']), tf.unstack(y_true['bounding_box']),
                    tf.unstack(y_pred['label']), tf.unstack(y_pred['bounding_box']))
        loss = []
        for y_true_labels, y_true_bounding_box, y_pred_labels, y_pred_bounding_box in zip(*zip_args):
            non_padding_mask = tf.cast(1.0 - y_true_labels[..., -1], dtype=tf.bool)
            y_true_labels = tf.boolean_mask(y_true_labels, non_padding_mask, axis=0)
            y_true_bounding_box = tf.boolean_mask(y_true_bounding_box, non_padding_mask, axis=0)
            label_scores = tf.matmul(y_true_labels, y_pred_labels, transpose_b=True)
            bounding_box_scores = compute_iou(boxes1=y_true_bounding_box, boxes2=y_pred_bounding_box,
                                              bounding_box_format='rel_xyxy')
            cost_matrix = 2.0 - label_scores - bounding_box_scores
            assignment = tf.stop_gradient(tf_linear_sum_assignment(cost_matrix))
            loss.append(tf.math.reduce_mean(tf.gather_nd(cost_matrix, tf.where(assignment))))
        return tf.convert_to_tensor(loss, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_weight': self.label_weight,
            'bounding_box_weight': self.bounding_box_weight,
            'iou_weight': self.iou_weight
        })
        return config
