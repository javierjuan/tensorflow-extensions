import numpy as np
import tensorflow as tf
from keras_cv.bounding_box import compute_iou
from scipy.optimize import linear_sum_assignment

from ..ops.hungarian import hungarian_matching


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

        self._loss_metric = tf.keras.metrics.Mean(name='detr_loss')

    def call(self, y_true, y_pred):
        y_true_labels, y_true_bounding_box = y_true['label'], y_true['bounding_box']
        y_pred_labels, y_pred_bounding_box = y_pred['label'], y_pred['bounding_box']
        batch_size = y_true_labels.nrows() if isinstance(y_true_labels, tf.RaggedTensor) else tf.shape(y_true_labels)[0]
        padding_mask = tf.expand_dims(y_true_labels[..., -1], axis=-1)

        label_scores = tf.matmul(y_true_labels, y_pred_labels, transpose_b=True)
        bounding_box_scores = compute_iou(boxes1=y_true_bounding_box, boxes2=y_pred_bounding_box,
                                          bounding_box_format='rel_xyxy', use_masking=True, mask_val=0)
        cost_matrix = 2.0 - label_scores - bounding_box_scores

        # cost_matrix += (padding_mask * 100)
        assignment = tf.stack([tf.stop_gradient(tf_linear_sum_assignment(cm)) for cm in tf.unstack(cost_matrix)])
        # assignment = tf.stop_gradient(hungarian_matching(cost_matrix=cost_matrix))

        scores = cost_matrix * tf.cast(assignment, dtype=tf.float32)
        loss = (tf.math.reduce_sum(scores * (1.0 - padding_mask), axis=[1, 2]) /
                tf.math.reduce_sum((1.0 - padding_mask), axis=[1, 2]))
        self._loss_metric.update_state(loss, sample_weight=batch_size)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_weight': self.label_weight,
            'bounding_box_weight': self.bounding_box_weight,
            'iou_weight': self.iou_weight
        })
        return config
