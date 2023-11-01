import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment


@tf.numpy_function(Tout=tf.bool)
def linear_sum_assignment(cost_matrix):
    rows, cols = _linear_sum_assignment(cost_matrix)
    assignment = np.zeros_like(cost_matrix, dtype=bool)
    assignment[rows, cols] = True
    return assignment
