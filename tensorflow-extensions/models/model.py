import warnings

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package='tfe.models')
class Model(keras.Model):
    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            with tf.name_scope(name='Forward'):
                if self._call_has_training_arg:
                    y_pred = self(x, training=True)
                else:
                    y_pred = self(x)
                with tf.name_scope('Loss'):
                    loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
                    self._loss_tracker.update_state(loss)
                    if self.optimizer is not None:
                        loss = self.optimizer.scale_loss(loss)

        with tf.name_scope(name='Backward'):
            if self.trainable_weights:
                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            else:
                warnings.warn('The model does not have any trainable weights.')

        with tf.name_scope(name='Metrics'):
            return self.compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
