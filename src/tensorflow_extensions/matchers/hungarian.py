import math

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment


@tf.numpy_function(Tout=tf.bool)
def linear_sum_assignment(cost_matrix):
    rows, cols = _linear_sum_assignment(cost_matrix)
    assignment = np.zeros_like(cost_matrix, dtype=bool)
    assignment[rows, cols] = True
    return assignment


def compute_ciou(boxes1, boxes2):
    y_min1, x_min1, y_max1, x_max1 = tf.split(boxes1[..., :4], 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(boxes2[..., :4], 4, axis=-1)

    boxes2_rank = len(boxes2.shape)
    perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]

    intersect_ymax = tf.math.minimum(y_max1, tf.transpose(y_max2, perm))
    intersect_ymin = tf.math.maximum(y_min1, tf.transpose(y_min2, perm))
    intersect_xmax = tf.math.minimum(x_max1, tf.transpose(x_max2, perm))
    intersect_xmin = tf.math.maximum(x_min1, tf.transpose(x_min2, perm))

    intersect_height = intersect_ymax - intersect_ymin
    intersect_width = intersect_xmax - intersect_xmin
    zero = tf.cast(0, intersect_height.dtype)
    intersect_height = tf.math.maximum(zero, intersect_height)
    intersect_width = tf.math.maximum(zero, intersect_width)
    intersect_area = intersect_height * intersect_width

    boxes1_area = tf.squeeze((y_max1 - y_min1) * (x_max1 - x_min1), axis=-1)
    boxes2_area = tf.squeeze((y_max2 - y_min2) * (x_max2 - x_min2), axis=-1)
    boxes2_area_rank = len(boxes2_area.shape)
    boxes2_axis = 1 if (boxes2_area_rank == 2) else 0
    boxes1_area = tf.expand_dims(boxes1_area, axis=-1)
    boxes2_area = tf.expand_dims(boxes2_area, axis=boxes2_axis)
    union_area = boxes1_area + boxes2_area - intersect_area
    iou = tf.math.divide(intersect_area, union_area + tf.keras.backend.epsilon())
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)

    convex_width = (tf.math.maximum(x_max1, tf.transpose(x_max2, perm)) -
                    tf.math.minimum(x_min1, tf.transpose(x_min2, perm)))
    convex_height = (tf.math.maximum(y_max1, tf.transpose(y_max2, perm)) -
                     tf.math.minimum(y_min1, tf.transpose(y_min2, perm)))
    convex_diagonal_squared = convex_width ** 2 + convex_height ** 2 + tf.keras.backend.epsilon()
    centers_distance_squared = (((x_min1 + x_max1) / 2 - tf.transpose((x_min2 + x_max2) / 2, perm)) ** 2 +
                                ((y_min1 + y_max1) / 2 - tf.transpose((y_min2 + y_max2) / 2, perm)) ** 2)

    width_1 = x_max1 - x_min1
    height_1 = y_max1 - y_min1 + tf.keras.backend.epsilon()
    width_2 = x_max2 - x_min2
    height_2 = y_max2 - y_min2 + tf.keras.backend.epsilon()

    v = tf.math.pow((4 / math.pi ** 2) * (tf.transpose(tf.math.atan(width_2 / height_2), perm) -
                                          tf.math.atan(width_1 / height_1)), 2)
    alpha = v / (v - iou + (1 + tf.keras.backend.epsilon()))

    return iou - (centers_distance_squared / convex_diagonal_squared + v * alpha)


def compute_matching(y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box, label_weight=1.0,
                     bounding_box_weight=1.0, padding_axis=-1):
    non_padding_mask = 1.0 - tf.expand_dims(y_true_label[..., padding_axis], axis=-1)
    label_scores = (1.0 - tf.matmul(y_true_label, y_pred_label, transpose_b=True)) * non_padding_mask
    bounding_box_scores = (1.0 - compute_ciou(y_true_bounding_box, y_pred_bounding_box)) * non_padding_mask
    cost_matrix = label_weight * label_scores + bounding_box_weight * bounding_box_scores
    return linear_sum_assignment(cost_matrix)


class HungarianMatcher(tf.Module):
    def __init__(self,
                 label_weight=1.0,
                 bounding_box_weight=1.0,
                 padding_axis=-1,
                 name=None):
        super().__init__(name=name)
        self.label_weight = label_weight
        self.bounding_box_weight = bounding_box_weight
        self.padding_axis = padding_axis

    def __call__(self, y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box):
        with self.name_scope:
            return tf.stop_gradient(
                compute_matching(y_true_label=y_true_label, y_true_bounding_box=y_true_bounding_box,
                                 y_pred_label=y_pred_label, y_pred_bounding_box=y_pred_bounding_box,
                                 label_weight=self.label_weight, bounding_box_weight=self.bounding_box_weight,
                                 padding_axis=self.padding_axis))
