import tensorflow as tf


@tf.function
def _label_smoothing(y_true, label_smoothing):
    num_labels = tf.cast(tf.shape(y_true)[-1], tf.float32)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_labels)


@tf.function
def initialize_loss(y_true, y_pred, label_smoothing=0.1, from_logits=False):
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), dtype=tf.float32)
    y_true = tf.cast(tf.convert_to_tensor(y_true), dtype=tf.float32)
    label_smoothing = tf.cast(tf.convert_to_tensor(label_smoothing), dtype=tf.float32)

    if not y_true.shape.is_compatible_with(y_pred.shape):
        raise ValueError(f'Incompatible shapes for y_true: {y_true.shape} and y_pred: {y_pred.shape}')

    y_pred = tf.math.softmax(logits=y_pred) if from_logits else y_pred

    # Perform ALWAYS a small label smoothing to prevent division by zero in metrics. This is better to add an epsilon
    # value at the numerator and denominator of each metric
    label_smoothing = 0.1 if label_smoothing < 0.1 else label_smoothing
    y_true = _label_smoothing(y_true=y_true, label_smoothing=label_smoothing)

    return y_true, y_pred


@tf.function
def finalize_loss(loss, reduction='sum', label_penalties=None):
    if label_penalties is not None:
        label_penalties = 1.0 + tf.cast(tf.convert_to_tensor(label_penalties), dtype=tf.float32)
        if not label_penalties.shape.is_compatible_with(loss.shape):
            label_penalties = tf.broadcast_to(label_penalties, loss.shape)
        loss = tf.multiply(loss, label_penalties)

    if reduction == 'sum':
        loss = tf.reduce_sum(loss, axis=-1)
    else:
        loss = tf.reduce_mean(loss, axis=-1)

    return loss