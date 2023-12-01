import keras_core as keras
from keras_core import ops


@keras.saving.register_keras_serializable(package='tfe.layers')
class PatchExtractor2D(keras.layers.Layer):
    def __init__(self,
                 size,
                 strides=None,
                 dilation_rate=1,
                 padding='same',
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if len(size) != 2:
            raise ValueError(f'`size` must be an array-like of size 2. Got: {size}')
        if strides is not None and len(strides) != 2:
            raise ValueError(f'`stride` must be an array-like of size 2. Got: {strides}')

        self.size = size
        self.strides = self.size if strides is None else strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return ops.image.extract_patches(image=inputs, size=self.size, strides=self.strides,
                                         dilation_rate=self.dilation_rate, padding=self.padding)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // self.strides[0], input_shape[2] // self.strides[1],
                self.strides[0] * self.strides[1] * input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'padding': self.padding
        })
        return config
