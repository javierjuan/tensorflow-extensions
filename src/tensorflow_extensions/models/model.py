import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package='tfe.models')
class Model(tf.keras.Model):
    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            with tf.name_scope(name='Forward'):
                y_pred = self(x, training=True)
                with tf.name_scope('Loss'):
                    loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)

        with tf.name_scope(name='Backward'):
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        with tf.name_scope(name='Metrics'):
            return self.compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
