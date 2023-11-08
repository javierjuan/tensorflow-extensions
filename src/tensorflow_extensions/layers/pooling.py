import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class ChannelAveragePooling(tf.keras.layers.Layer):
    def __init__(self,
                 keepdims=False,
                 data_format=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.keepdims = keepdims
        self.data_format = tf.keras.backend.image_data_format().lower() if data_format is None else data_format.lower()
        if self.data_format not in ('channels_first', 'channels_last'):
            raise ValueError('The `data_format` argument must be one of "channels_first", "channels_last". '
                             f'Received: {self.data_format}')

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            return tf.math.reduce_mean(inputs, axis=-1, keepdims=self.keepdims)
        else:
            return tf.math.reduce_mean(inputs, axis=1, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({
            'keepdims': self.keepdims,
            'data_format': self.data_format
        })
        return config


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class ChannelMaxPooling(tf.keras.layers.Layer):
    def __init__(self,
                 keepdims=False,
                 data_format=None,
                 name=None,
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.keepdims = keepdims
        self.data_format = tf.keras.backend.image_data_format().lower() if data_format is None else data_format.lower()
        if self.data_format not in ('channels_first', 'channels_last'):
            raise ValueError('The `data_format` argument must be one of "channels_first", "channels_last". '
                             f'Received: {self.data_format}')

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            return tf.math.reduce_max(inputs, axis=-1, keepdims=self.keepdims)
        else:
            return tf.math.reduce_max(inputs, axis=1, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({
            'keepdims': self.keepdims,
            'data_format': self.data_format
        })
        return config
