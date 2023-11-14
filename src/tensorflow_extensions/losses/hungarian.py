import tensorflow as tf

from ..matchers import utils, Hungarian as HungarianMatcher
from ..matchers.hungarian import compute_iou, compute_distance


@tf.keras.saving.register_keras_serializable(package='tfe.losses')
class Hungarian(tf.keras.losses.Loss):
    def __init__(self, label_weight=1.0, bounding_box_weight=1.0, padding_axis=-1, focal=True, generalized=True, norm=1,
                 reduction=tf.keras.losses.Reduction.AUTO, name='hungarian'):
        super().__init__(name=name, reduction=reduction)
        self.label_weight = label_weight
        self.bounding_box_weight = bounding_box_weight
        self.padding_axis = padding_axis
        self.focal = focal
        self.generalized = generalized
        self.norm = norm
        self.matcher = HungarianMatcher(generalized=generalized, norm=norm, name='hungarian_matcher')

    @staticmethod
    def compute_label_loss(y_true, y_pred, focal=True, label_smoothing=0.0):
        if focal:
            loss = tf.keras.losses.categorical_focal_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False,
                                                                  label_smoothing=label_smoothing)
        else:
            loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False,
                                                            label_smoothing=label_smoothing)
        return tf.math.reduce_mean(loss)

    @staticmethod
    def compute_bounding_box_loss(y_true, y_pred, generalized=True, norm=1):
        iou = 1.0 - compute_iou(bounding_box_1=y_true, bounding_box_2=y_pred, generalized=generalized)
        distance = compute_distance(bounding_box_1=y_true, bounding_box_2=y_pred, norm=norm)
        loss = tf.linalg.tensor_diag_part(iou + distance)
        return tf.math.reduce_mean(loss)

    def call(self, y_true, y_pred):
        zip_args = (tf.unstack(y_true['label']), tf.unstack(y_true['bounding_box']),
                    tf.unstack(y_pred['label']), tf.unstack(y_pred['bounding_box']))
        batch_loss = []
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
                # Compute label loss based on Hungarian matching
                y_true_label = utils.index_by_assignment(y_true_label, assignment=assignment)
                label_loss = self.compute_label_loss(y_true=y_true_label, y_pred=y_pred_label, focal=self.focal)
                # Compute bounding box loss based on Hungarian matching
                o_pred_bounding_box = utils.gather_by_assignment(y_pred_bounding_box, assignment=assignment)
                bounding_box_loss = self.compute_bounding_box_loss(y_true=o_true_bounding_box,
                                                                   y_pred=o_pred_bounding_box,
                                                                   generalized=self.generalized, norm=self.norm)
                # Compute general loss
                loss = self.label_weight * label_loss + self.bounding_box_weight * bounding_box_loss
            else:
                loss = self.compute_label_loss(y_true=y_true_label, y_pred=y_pred_label)
            batch_loss.append(loss)
        return tf.convert_to_tensor(batch_loss, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_weight': self.label_weight,
            'bounding_box_weight': self.bounding_box_weight,
            'padding_axis': self.padding_axis,
            'generalized': self.generalized
        })
        return config
