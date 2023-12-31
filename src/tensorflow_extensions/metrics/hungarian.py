import tensorflow as tf

from ..matchers import utils, Hungarian as HungarianMatcher
from ..matchers.hungarian import compute_iou


@tf.keras.saving.register_keras_serializable(package='tfe.metrics')
class Hungarian(tf.keras.metrics.Metric):
    def __init__(self, padding_axis=-1, generalized=True, norm=1, name='hungarian', **kwargs):
        super().__init__(name=name, **kwargs)
        self.padding_axis = padding_axis
        self.generalized = generalized
        self.norm = norm

        self.matcher = HungarianMatcher(generalized=generalized, norm=norm, name='hungarian_matcher')
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.accuracy = self.add_weight(name='accuracy', initializer='zeros')
        self.objects = self.add_weight(name='objects', initializer='zeros')

    @staticmethod
    def compute_label_metric(y_true, y_pred):
        return tf.math.reduce_sum(tf.keras.metrics.categorical_accuracy(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def compute_bounding_box_metric(y_true, y_pred, generalized=True):
        scores = compute_iou(bounding_box_1=y_true, bounding_box_2=y_pred, generalized=generalized)
        scores = tf.linalg.tensor_diag_part(scores)
        return tf.math.reduce_sum(scores)

    def update_state(self, y_true, y_pred):
        zip_args = (tf.unstack(y_true['label']), tf.unstack(y_true['bounding_box']),
                    tf.unstack(y_pred['label']), tf.unstack(y_pred['bounding_box']))
        accuracy, iou, objects = [], [], []
        for y_true_label, y_true_bounding_box, y_pred_label, y_pred_bounding_box in zip(*zip_args):
            # Compute non-padding mask
            mask = utils.non_padding_mask(label=y_true_label, padding_axis=self.padding_axis)
            if tf.math.reduce_any(mask):
                # Compute non-padded labels and bounding boxes
                o_true_label = utils.gather_by_mask(y_true_label, mask=mask)
                o_true_bounding_box = utils.gather_by_mask(y_true_bounding_box, mask=mask)
                # Compute Hungarian matching
                assignment = self.matcher(y_true_label=o_true_label, y_true_bounding_box=o_true_bounding_box,
                                          y_pred_label=y_pred_label, y_pred_bounding_box=y_pred_bounding_box)
                # Compute label metric based on Hungarian matching
                o_pred_label = utils.gather_by_assignment(y_pred_label, assignment=assignment)
                label_metric = self.compute_label_metric(y_true=o_true_label, y_pred=o_pred_label)
                # Compute bounding box metric based on Hungarian matching
                o_pred_bounding_box = utils.gather_by_assignment(y_pred_bounding_box, assignment=assignment)
                bounding_box_metric = self.compute_bounding_box_metric(y_true=o_true_bounding_box,
                                                                       y_pred=o_pred_bounding_box,
                                                                       generalized=self.generalized)
            else:
                label_metric = tf.constant(value=0.0, dtype=tf.float32)
                bounding_box_metric = tf.constant(value=0.0, dtype=tf.float32)
            accuracy.append(label_metric)
            iou.append(bounding_box_metric)
            objects.append(tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32)))
        self.accuracy.assign_add(tf.math.reduce_sum(tf.convert_to_tensor(accuracy)))
        self.iou.assign_add(tf.math.reduce_sum(tf.convert_to_tensor(iou)))
        self.objects.assign_add(tf.math.reduce_sum(tf.convert_to_tensor(objects)))

    def result(self):
        return {'iou': tf.math.divide_no_nan(self.iou, self.objects),
                'accuracy': tf.math.divide_no_nan(self.accuracy, self.objects)}

    def reset_state(self):
        for variable in self.variables:
            variable.assign(tf.zeros(variable.shape, dtype=variable.dtype))

    def get_config(self):
        config = super().get_config()
        config.update({
            'padding_axis': self.padding_axis,
            'generalized': self.generalized,
            'norm': self.norm
        })
        return config
