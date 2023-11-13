import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment


@tf.numpy_function(Tout=tf.bool)
@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
def linear_sum_assignment(cost_matrix):
    rows, cols = _linear_sum_assignment(cost_matrix)
    assignment = np.zeros_like(cost_matrix, dtype=bool)
    assignment[rows, cols] = True
    return assignment


@tf.function
@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
def _permutation_indices(tensor):
    return tf.constant([1, 0]) if tf.rank(tensor) == 2 else tf.constant([0, 2, 1])


@tf.function
@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
def _compute_valid(bounding_box):
    y_min, x_min, y_max, x_max = tf.split(bounding_box[..., :4], 4, axis=-1)
    valid = tf.math.logical_and(tf.math.greater_equal(y_max, y_min), tf.math.greater_equal(x_max, x_min))
    return tf.cast(valid, dtype=bounding_box.dtype)


@tf.function
@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
def _compute_area(bounding_box):
    y_min, x_min, y_max, x_max = tf.split(bounding_box[..., :4], 4, axis=-1)
    area = (y_max - y_min) * (x_max - x_min)
    return tf.math.maximum(tf.cast(0, bounding_box.dtype), area)


@tf.function
@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
def _compute_intersection(bounding_box_1, bounding_box_2):
    y_min1, x_min1, y_max1, x_max1 = tf.split(bounding_box_1[..., :4], 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(bounding_box_2[..., :4], 4, axis=-1)
    perm = _permutation_indices(bounding_box_2)
    intersect_ymax = tf.math.minimum(y_max1, tf.transpose(y_max2, perm))
    intersect_ymin = tf.math.maximum(y_min1, tf.transpose(y_min2, perm))
    intersect_xmax = tf.math.minimum(x_max1, tf.transpose(x_max2, perm))
    intersect_xmin = tf.math.maximum(x_min1, tf.transpose(x_min2, perm))
    zero = tf.cast(0, bounding_box_1.dtype)
    intersect_height = tf.math.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_width = tf.math.maximum(zero, intersect_xmax - intersect_xmin)
    return intersect_height * intersect_width


@tf.function
@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
def _compute_convex_hull(bounding_box_1, bounding_box_2):
    y_min1, x_min1, y_max1, x_max1 = tf.split(bounding_box_1[..., :4], 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(bounding_box_2[..., :4], 4, axis=-1)
    perm = _permutation_indices(bounding_box_2)
    convex_hull_ymin = tf.math.minimum(y_min1, tf.transpose(y_min2, perm))
    convex_hull_xmin = tf.math.minimum(x_min1, tf.transpose(x_min2, perm))
    convex_hull_ymax = tf.math.maximum(y_max1, tf.transpose(y_max2, perm))
    convex_hull_xmax = tf.math.maximum(x_max1, tf.transpose(x_max2, perm))
    zero = tf.cast(0, bounding_box_1.dtype)
    convex_hull_height = tf.math.maximum(zero, convex_hull_ymax - convex_hull_ymin)
    convex_hull_width = tf.math.maximum(zero, convex_hull_xmax - convex_hull_xmin)
    return convex_hull_height * convex_hull_width


# @tf.function
@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
def compute_giou(bounding_box_1, bounding_box_2, mode='giou'):
    # Compute valid masks
    valid1 = _compute_valid(bounding_box_1)
    valid2 = _compute_valid(bounding_box_2)
    # Compute areas
    bounding_box_1_area = _compute_area(bounding_box_1) * valid1
    bounding_box_2_area = _compute_area(bounding_box_2) * valid2
    # Compute valid matrix
    perm = _permutation_indices(bounding_box_2_area)
    valid = tf.cast(tf.math.multiply(valid1, tf.transpose(valid2, perm)), bounding_box_1.dtype)
    # Compute iou
    intersect_area = _compute_intersection(bounding_box_1, bounding_box_2) * valid
    union_area = (bounding_box_1_area + tf.transpose(bounding_box_2_area, perm)) - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == 'iou':
        return iou
    convex_hull_area = _compute_convex_hull(bounding_box_1, bounding_box_2) * valid
    giou = iou - tf.math.divide_no_nan(convex_hull_area - union_area, convex_hull_area)
    return giou


# @tf.function
@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
def compute_matching(y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box, mode='giou'):
    label_scores = 1.0 - tf.linalg.matmul(y_true_label, y_pred_label, transpose_b=True, a_is_sparse=True)
    bounding_box_scores = 1.0 - compute_giou(y_true_bounding_box, y_pred_bounding_box, mode)
    cost_matrix = label_scores + bounding_box_scores
    return linear_sum_assignment(cost_matrix)


@tf.keras.saving.register_keras_serializable(package='tfe.matchers')
class Hungarian(tf.Module):
    def __init__(self, mode='giou', name='hungarian'):
        super().__init__(name=name)
        self.mode = mode

    def __call__(self, y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box, padding_axis=None):
        with self.name_scope:
            return tf.stop_gradient(
                compute_matching(y_true_label=y_true_label, y_true_bounding_box=y_true_bounding_box,
                                 y_pred_label=y_pred_label, y_pred_bounding_box=y_pred_bounding_box, mode=self.mode))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            'mode': self.mode
        }
