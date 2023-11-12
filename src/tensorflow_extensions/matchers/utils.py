import tensorflow as tf


@tf.function
def index_by_assignment(tensor, assignment):
    assigned_indices = tf.where(assignment)[:, -1]
    unassigned_indices = tf.where(tf.math.reduce_all(tf.math.logical_not(assignment), axis=0))[:, -1]
    indices = tf.argsort(tf.concat([assigned_indices, unassigned_indices], axis=0))
    return tf.gather(tensor, indices, axis=0)


@tf.function
def gather_by_assignment(tensor, assignment):
    indices = tf.where(assignment)[:, -1]
    return tf.gather(tensor, indices, axis=0)


@tf.function
def gather_by_mask(tensor, mask):
    return tf.boolean_mask(tensor, mask=mask, axis=0)


@tf.function
def padding_mask(label, padding_axis=-1):
    return tf.cast(label[..., padding_axis], dtype=tf.bool)


@tf.function
def non_padding_mask(label, padding_axis=-1):
    return tf.math.logical_not(padding_mask(label=label, padding_axis=padding_axis))
