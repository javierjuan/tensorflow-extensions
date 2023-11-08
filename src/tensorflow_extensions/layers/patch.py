import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package='tfe.layers')
class PatchExtractor2D(tf.keras.layers.Layer):
    def __init__(self,
                 size,
                 strides=None,
                 rates=(1, 1),
                 padding='valid',
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if len(size) != 2:
            raise ValueError(f'`size` must be an array-like of size 2. Got: {size}')
        if strides is not None and len(strides) != 2:
            raise ValueError(f'`stride` must be an array-like of size 2. Got: {strides}')
        if rates is not None and len(rates) != 2:
            raise ValueError(f'`rate` must be an array-like of size 2. Got: {rates}')

        self.size = [1, *size, 1]
        self.strides = self.size if strides is None else [1, *strides, 1]
        self.rates = [1, 1, 1, 1] if rates is None else [1, *rates, 1]
        self.padding = padding
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return tf.image.extract_patches(images=inputs, sizes=self.size, strides=self.strides, rates=self.rates,
                                        padding=self.padding.upper())

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size[1:-1],
            'strides': self.strides[1:-1],
            'rates': self.rates[1:-1],
            'padding': self.padding
        })
        return config
