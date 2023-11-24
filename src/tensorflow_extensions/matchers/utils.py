import tensorflow as tf
from keras_core import ops


@tf.function(reduce_retracing=True)
def index_by_assignment(tensors, assignment):
    assigned_indices = ops.where(assignment)[-1]
    unassigned_indices = ops.where(ops.all(ops.logical_not(assignment), axis=0))[-1]
    indices = ops.argsort(ops.concatenate([assigned_indices, unassigned_indices], axis=0))
    return [ops.take(tensor, indices, axis=0) for tensor in tensors]


@tf.function(reduce_retracing=True)
def gather_by_assignment(tensor, assignment):
    indices = ops.where(assignment)[-1]
    return ops.take(tensor, indices, axis=0)


@tf.function(reduce_retracing=True)
def gather_by_mask(tensor, mask):
    return ops.take(tensor, indices=ops.where(mask)[0], axis=0)


@tf.function(reduce_retracing=True)
def padding_mask(label, padding_axis=-1):
    return ops.cast(label[..., padding_axis], dtype='bool')


@tf.function(reduce_retracing=True)
def non_padding_mask(label, padding_axis=-1):
    return ops.logical_not(padding_mask(label=label, padding_axis=padding_axis))
