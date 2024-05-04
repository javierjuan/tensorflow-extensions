import keras
import tensorflow as tf
from keras import ops

from ..matchers import utils
from ..matchers.hungarian import compute_matching, compute_iou


@keras.saving.register_keras_serializable(package='tfe.metrics')
class Hungarian(keras.metrics.Metric):
    def __init__(self, padding_axis=-1, generalized=True, order=1, name='hungarian', **kwargs):
        super().__init__(name=name, **kwargs)
        self.padding_axis = padding_axis
        self.generalized = generalized
        self.order = order

        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.accuracy = self.add_weight(name='accuracy', initializer='zeros')
        self.objects = self.add_weight(name='objects', initializer='zeros')

    @staticmethod
    def compute_label_metric(y_true, y_pred, mask):
        scores = keras.metrics.categorical_accuracy(y_true=y_true, y_pred=y_pred)
        return ops.sum(scores * ops.cast(mask, scores.dtype))

    @staticmethod
    def compute_bounding_box_metric(y_true, y_pred, mask, generalized=True):
        scores = compute_iou(bounding_box_1=y_true, bounding_box_2=y_pred, generalized=generalized)
        scores = ops.diag(scores)
        return ops.sum(scores * ops.cast(mask, scores.dtype))

    @tf.function
    def update_state(self, y_true, y_pred):
        zip_args = (ops.unstack(y_true['label']), ops.unstack(y_true['bounding_box']),
                    ops.unstack(y_pred['label']), ops.unstack(y_pred['bounding_box']))
        accuracy, iou, objects = [], [], []
        for y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box in zip(*zip_args):
            # Compute non-padding mask
            mask = utils.non_padding_mask(label=y_true_label, padding_axis=self.padding_axis)
            if ops.any(mask):
                # Compute non-padded labels and bounding boxes
                o_true_label = utils.gather_by_mask(y_true_label, mask=mask)
                o_true_bounding_box = utils.gather_by_mask(y_true_bounding_box, mask=mask)
                # Compute Hungarian matching
                assignment = ops.stop_gradient(
                    compute_matching(y_true_label=o_true_label, y_true_bounding_box=o_true_bounding_box,
                                     y_pred_label=y_pred_label, y_pred_bounding_box=y_pred_bounding_box))
                # Compute label metric based on Hungarian matching
                output = utils.index_by_assignment(tensors=[y_true_label, y_true_bounding_box, mask],
                                                   assignment=assignment)
                y_true_label, y_true_bounding_box, mask = output
                label_metric = self.compute_label_metric(y_true=y_true_label, y_pred=y_pred_label, mask=mask)
                # Compute bounding box metric based on Hungarian matching
                bounding_box_metric = self.compute_bounding_box_metric(y_true=y_true_bounding_box,
                                                                       y_pred=y_pred_bounding_box, mask=mask,
                                                                       generalized=self.generalized)
            else:
                label_metric = tf.constant(value=0.0, dtype=tf.float32)
                bounding_box_metric = tf.constant(value=0.0, dtype=tf.float32)
            accuracy.append(label_metric)
            iou.append(bounding_box_metric)
            objects.append(ops.sum(ops.cast(mask, dtype='float32')))
        self.accuracy.assign_add(ops.sum(ops.convert_to_tensor(accuracy)))
        self.iou.assign_add(ops.sum(ops.convert_to_tensor(iou)))
        self.objects.assign_add(ops.sum(ops.convert_to_tensor(objects)))

    def result(self):
        return {'iou': self.iou / self.objects,
                'accuracy': self.accuracy / self.objects}

    def reset_state(self):
        for variable in self.variables:
            variable.assign(ops.zeros(variable.shape, dtype=variable.dtype))

    def get_config(self):
        config = super().get_config()
        config.update({
            'padding_axis': self.padding_axis,
            'generalized': self.generalized,
            'order': self.order
        })
        return config
