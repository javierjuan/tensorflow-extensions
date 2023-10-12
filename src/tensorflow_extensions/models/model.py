import tensorflow as tf


class Model(tf.keras.Model):
    def train_step(self, data):
        x, y, sample_weight = data if len(data) == 3 else data + (None,)

        with tf.GradientTape() as tape:
            with tf.name_scope(name='Forward'):
                y_pred = self(x, training=True)
                with tf.name_scope('Loss'):
                    loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        with tf.name_scope(name='Backward'):
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        with tf.name_scope(name='Metrics'):
            self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
            return {metric.name: metric.result() for metric in self.metrics}
