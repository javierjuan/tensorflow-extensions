import keras
import numpy as np
import tensorflow as tf
from keras import ops
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

from ..ops import math


@tf.py_function(Tout=tf.bool)
def linear_sum_assignment(cost_matrix):
    rows, cols = _linear_sum_assignment(cost_matrix)
    assignment = np.zeros_like(cost_matrix, dtype=bool)
    assignment[rows, cols] = True
    return assignment


@tf.function(reduce_retracing=True)
def _compute_valid(bounding_box):
    y_min, x_min, y_max, x_max = ops.split(bounding_box, indices_or_sections=4, axis=-1)
    return ops.logical_and(ops.greater(y_max, y_min), ops.greater(x_max, x_min))


@tf.function(reduce_retracing=True)
def _compute_area(bounding_box):
    y_min, x_min, y_max, x_max = ops.split(bounding_box, indices_or_sections=4, axis=-1)
    area = (y_max - y_min) * (x_max - x_min)
    return ops.maximum(ops.cast(0, area.dtype), area)


@tf.function(reduce_retracing=True)
def _compute_intersection(bounding_box_1, bounding_box_2):
    y_min1, x_min1, y_max1, x_max1 = ops.split(bounding_box_1, indices_or_sections=4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = ops.split(bounding_box_2, indices_or_sections=4, axis=-1)
    axes = [1, 0] if ops.ndim(bounding_box_2) == 2 else [0, 2, 1]
    intersect_ymax = ops.minimum(y_max1, ops.transpose(y_max2, axes))
    intersect_ymin = ops.maximum(y_min1, ops.transpose(y_min2, axes))
    intersect_xmax = ops.minimum(x_max1, ops.transpose(x_max2, axes))
    intersect_xmin = ops.maximum(x_min1, ops.transpose(x_min2, axes))
    zero = ops.cast(0, bounding_box_1.dtype)
    intersect_height = ops.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_width = ops.maximum(zero, intersect_xmax - intersect_xmin)
    intersection = intersect_height * intersect_width
    return ops.maximum(ops.cast(0, intersection.dtype), intersection)


@tf.function(reduce_retracing=True)
def _compute_convex_hull(bounding_box_1, bounding_box_2):
    y_min1, x_min1, y_max1, x_max1 = ops.split(bounding_box_1, indices_or_sections=4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = ops.split(bounding_box_2, indices_or_sections=4, axis=-1)
    axes = [1, 0] if ops.ndim(bounding_box_2) == 2 else [0, 2, 1]
    convex_hull_ymin = ops.minimum(y_min1, ops.transpose(y_min2, axes))
    convex_hull_xmin = ops.minimum(x_min1, ops.transpose(x_min2, axes))
    convex_hull_ymax = ops.maximum(y_max1, ops.transpose(y_max2, axes))
    convex_hull_xmax = ops.maximum(x_max1, ops.transpose(x_max2, axes))
    zero = ops.cast(0, bounding_box_1.dtype)
    convex_hull_height = ops.maximum(zero, convex_hull_ymax - convex_hull_ymin)
    convex_hull_width = ops.maximum(zero, convex_hull_xmax - convex_hull_xmin)
    convex_hull = convex_hull_height * convex_hull_width
    return ops.maximum(ops.cast(0, convex_hull.dtype), convex_hull)


@tf.function(reduce_retracing=True)
def compute_iou(bounding_box_1, bounding_box_2, generalized=True):
    valid1 = ops.cast(_compute_valid(bounding_box_1), bounding_box_1.dtype)
    valid2 = ops.cast(_compute_valid(bounding_box_2), bounding_box_2.dtype)
    bounding_box_1_area = _compute_area(bounding_box_1)
    bounding_box_2_area = _compute_area(bounding_box_2)
    axes = [1, 0] if ops.ndim(bounding_box_2_area) == 2 else [0, 2, 1]
    valid = valid1 * ops.transpose(valid2, axes)
    intersection = _compute_intersection(bounding_box_1, bounding_box_2)
    union = (bounding_box_1_area + ops.transpose(bounding_box_2_area, axes)) - intersection
    iou = ops.where(valid, intersection / union, ops.zeros_like(intersection))
    if generalized:
        convex_hull = _compute_convex_hull(bounding_box_1, bounding_box_2)
        iou = ops.where(valid, iou - (convex_hull - union) / convex_hull, -ops.ones_like(iou))
    return iou


@tf.function(reduce_retracing=True)
def compute_distance(bounding_box_1, bounding_box_2, order=1):
    valid1 = ops.cast(_compute_valid(bounding_box_1), bounding_box_1.dtype)
    valid2 = ops.cast(_compute_valid(bounding_box_2), bounding_box_2.dtype)
    axes = [1, 0] if ops.ndim(bounding_box_2) == 2 else [0, 2, 1]
    valid = valid1 * ops.transpose(valid2, axes)
    distance = ops.expand_dims(bounding_box_1, axis=-2) - ops.expand_dims(bounding_box_2, axis=-3)
    distance = math.norm(distance, order=order, axis=-1) / math.norm(ops.ones(shape=4), order=order)
    return ops.where(valid, distance, ops.ones_like(distance))


@tf.function(reduce_retracing=True)
def compute_matching(y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box, label_weight=1.0,
                     iou_weight=2.0, distance_weight=5.0, generalized=True, order=1):
    label_scores = 1.0 - ops.matmul(y_true_label, ops.transpose(y_pred_label))
    iou_scores = 1.0 - compute_iou(y_true_bounding_box, y_pred_bounding_box, generalized=generalized)
    distance_scores = compute_distance(y_true_bounding_box, y_pred_bounding_box, order=order)
    cost_matrix = label_weight * label_scores + iou_weight * iou_scores + distance_weight * distance_scores
    assignment = linear_sum_assignment(cost_matrix)
    assignment.set_shape(cost_matrix.shape)
    return assignment


@keras.saving.register_keras_serializable(package='tfe.matchers')
class Hungarian(tf.Module):
    def __init__(self, label_weight=1.0, iou_weight=2.0, distance_weight=5.0, generalized=True, order=1,
                 name='hungarian'):
        super().__init__(name=name)
        self.label_weight = label_weight
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        self.generalized = generalized
        self.order = order

    @tf.Module.with_name_scope
    def __call__(self, y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box, padding_axis=None):
        return ops.stop_gradient(
            compute_matching(y_true_label=y_true_label, y_true_bounding_box=y_true_bounding_box,
                             y_pred_label=y_pred_label, y_pred_bounding_box=y_pred_bounding_box,
                             label_weight=self.label_weight, iou_weight=self.iou_weight,
                             distance_weight=self.distance_weight, generalized=self.generalized, order=self.order))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            'label_weight': self.label_weight,
            'iou_weight': self.iou_weight,
            'distance_weight': self.distance_weight,
            'generalized': self.generalized,
            'order': self.order
        }
