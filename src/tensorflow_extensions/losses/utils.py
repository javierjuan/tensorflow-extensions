import keras_core as keras
import tensorflow as tf
from keras_core import ops


@tf.function
@keras.saving.register_keras_serializable(package='tfe.losses')
def _label_smoothing(y_true, label_smoothing):
    num_labels = ops.cast(ops.shape(y_true)[-1], dtype='float32')
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_labels)


@tf.function
@keras.saving.register_keras_serializable(package='tfe.losses')
def initialize_loss(y_true, y_pred, label_smoothing=0.1, from_logits=False):
    y_pred = ops.convert_to_tensor(y_pred, dtype='float32')
    y_true = ops.convert_to_tensor(y_true, dtype='float32')
    label_smoothing = ops.convert_to_tensor(label_smoothing, dtype='float32')

    if not y_true.shape.is_compatible_with(y_pred.shape):
        raise ValueError(f'Incompatible shapes for y_true: {y_true.shape} and y_pred: {y_pred.shape}')

    y_pred = ops.softmax(logits=y_pred) if from_logits else y_pred

    # Perform ALWAYS a small label smoothing to prevent division by zero in metrics. This is better to add an epsilon
    # value at the numerator and denominator of each metric
    label_smoothing = 0.1 if label_smoothing < 0.1 else label_smoothing
    y_true = _label_smoothing(y_true=y_true, label_smoothing=label_smoothing)

    return y_true, y_pred


@tf.function
@keras.saving.register_keras_serializable(package='tfe.losses')
def finalize_loss(loss, reduction='sum', label_penalties=None):
    if label_penalties is not None:
        label_penalties = 1.0 + ops.convert_to_tensor(label_penalties, dtype='float32')
        if not label_penalties.shape.is_compatible_with(loss.shape):
            label_penalties = ops.broadcast_to(label_penalties, loss.shape)
        loss *= label_penalties

    if reduction == 'sum':
        loss = ops.sum(loss, axis=-1)
    else:
        loss = ops.mean(loss, axis=-1)

    return loss
