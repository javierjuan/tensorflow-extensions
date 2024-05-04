import keras
import tensorflow as tf
from keras import ops

from ..matchers import utils, Hungarian as HungarianMatcher
from ..matchers.hungarian import compute_matching, compute_iou, compute_distance


@tf.function(reduce_retracing=True)
def compute_label_loss(y_true, y_pred, mask, focal=True, empty_weight=0.1, label_smoothing=0.0):
    if focal:
        loss = keras.losses.categorical_focal_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False,
                                                           label_smoothing=label_smoothing)
    else:
        loss = keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False,
                                                     label_smoothing=label_smoothing)
    return ops.mean(ops.where(mask, loss, loss * empty_weight))


@tf.function(reduce_retracing=True)
def compute_iou_loss(y_true, y_pred, mask, generalized=False):
    iou = 1.0 - compute_iou(bounding_box_1=y_true, bounding_box_2=y_pred, generalized=generalized)
    loss = ops.diag(iou)
    mask = ops.cast(mask, loss.dtype)
    return ops.sum(loss * mask) / ops.sum(mask)


@tf.function(reduce_retracing=True)
def compute_distance_loss(y_true, y_pred, mask, order=1):
    distance = compute_distance(bounding_box_1=y_true, bounding_box_2=y_pred, order=order)
    loss = ops.diag(distance)
    mask = ops.cast(mask, loss.dtype)
    return ops.sum(loss * mask) / ops.sum(mask)


@tf.function(reduce_retracing=True)
def hungarian_loss(y_true, y_pred):
    zip_args = (ops.unstack(y_true['label']), ops.unstack(y_true['bounding_box']),
                ops.unstack(y_pred['label']), ops.unstack(y_pred['bounding_box']))
    batch_loss = []
    for y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box in zip(*zip_args):
        # Compute non-padding mask
        mask = utils.non_padding_mask(label=y_true_label, padding_axis=-1)
        if ops.any(mask):
            # Compute non-padded labels and bounding boxes
            o_true_label = utils.gather_by_mask(y_true_label, mask=mask)
            o_true_bounding_box = utils.gather_by_mask(y_true_bounding_box, mask=mask)
            # Compute Hungarian matching
            assignment = ops.stop_gradient(
                compute_matching(y_true_label=o_true_label, y_true_bounding_box=o_true_bounding_box,
                                 y_pred_label=y_pred_label, y_pred_bounding_box=y_pred_bounding_box))
            # Compute label loss based on Hungarian matching
            output = utils.index_by_assignment(tensors=[y_true_label, y_true_bounding_box, mask],
                                               assignment=assignment)
            y_true_label, y_true_bounding_box, mask = output
            label_loss = compute_label_loss(y_true=y_true_label, y_pred=y_pred_label, mask=mask)
            # Compute bounding box loss based on Hungarian matching
            iou_loss = compute_iou_loss(y_true=y_true_bounding_box, y_pred=y_pred_bounding_box, mask=mask)
            # Compute distance loss based on Hungarian matching
            distance_loss = compute_distance_loss(y_true=y_true_bounding_box, y_pred=y_pred_bounding_box, mask=mask)
            # Compute general loss
            loss = (1.0 * label_loss + 2.0 * iou_loss + 5.0 * distance_loss)
        else:
            loss = compute_label_loss(y_true=y_true_label, y_pred=y_pred_label, mask=mask)
        batch_loss.append(loss)
    return ops.mean(ops.convert_to_tensor(batch_loss, dtype=keras.backend.floatx()))


@keras.saving.register_keras_serializable(package='tfe.losses')
class Hungarian(keras.losses.Loss):
    def __init__(self, label_weight=1.0, iou_weight=2.0, distance_weight=5.0, padding_axis=-1, focal=True,
                 generalized=True, order=1, empty_weight=0.1, reduction='sum_over_batch_size', name='hungarian'):
        super().__init__(name=name, reduction=reduction)
        self.label_weight = label_weight
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        self.padding_axis = padding_axis
        self.focal = focal
        self.generalized = generalized
        self.order = order
        self.empty_weight = empty_weight
        self.matcher = HungarianMatcher(generalized=generalized, order=order, name='hungarian_matcher')

    def call(self, y_true, y_pred):
        zip_args = (ops.unstack(y_true['label']), ops.unstack(y_true['bounding_box']),
                    ops.unstack(y_pred['label']), ops.unstack(y_pred['bounding_box']))
        batch_loss = []
        for y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box in zip(*zip_args):
            # Compute non-padding mask
            mask = utils.non_padding_mask(label=y_true_label, padding_axis=self.padding_axis)
            if ops.any(mask):
                # Compute non-padded labels and bounding boxes
                o_true_label = utils.gather_by_mask(y_true_label, mask=mask)
                o_true_bounding_box = utils.gather_by_mask(y_true_bounding_box, mask=mask)
                # Compute Hungarian matching
                assignment = self.matcher(y_true_label=o_true_label, y_true_bounding_box=o_true_bounding_box,
                                          y_pred_label=y_pred_label, y_pred_bounding_box=y_pred_bounding_box)
                # Compute label loss based on Hungarian matching
                output = utils.index_by_assignment(tensors=[y_true_label, y_true_bounding_box, mask],
                                                   assignment=assignment)
                y_true_label, y_true_bounding_box, mask = output
                label_loss = compute_label_loss(y_true=y_true_label, y_pred=y_pred_label, mask=mask,
                                                empty_weight=self.empty_weight, focal=self.focal)
                # Compute bounding box loss based on Hungarian matching
                iou_loss = compute_iou_loss(y_true=y_true_bounding_box, y_pred=y_pred_bounding_box, mask=mask,
                                            generalized=self.generalized)
                distance_loss = compute_distance_loss(y_true=y_true_bounding_box, y_pred=y_pred_bounding_box,
                                                      mask=mask, order=self.order)
                # Compute general loss
                loss = (self.label_weight * label_loss + self.iou_weight * iou_loss + self.distance_weight *
                        distance_loss)
            else:
                loss = compute_label_loss(y_true=y_true_label, y_pred=y_pred_label, mask=mask)
            batch_loss.append(loss)
        return ops.convert_to_tensor(batch_loss, dtype='float32')

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_weight': self.label_weight,
            'iou_weight': self.iou_weight,
            'distance_weight': self.distance_weight,
            'padding_axis': self.padding_axis,
            'focal': self.focal,
            'generalized': self.generalized,
            'order': self.order,
            'empty_weight': self.empty_weight
        })
        return config
