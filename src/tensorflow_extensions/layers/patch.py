import keras_core as keras
from keras_core import ops


@keras.saving.register_keras_serializable(package='tfe.layers')
class PatchExtractor2D(keras.layers.Layer):
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
        # TODO: Check this
        return ops.image.extract_patches(image=inputs, size=self.size, strides=self.strides, dilation_rate=self.rates,
                                         padding=self.padding)

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size[1:-1],
            'strides': self.strides[1:-1],
            'rates': self.rates[1:-1],
            'padding': self.padding
        })
        return config
