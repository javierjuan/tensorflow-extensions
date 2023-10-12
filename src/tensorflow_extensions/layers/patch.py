import tensorflow as tf


class PatchExtractor2D(tf.keras.layers.Layer):
    def __init__(self,
                 size,
                 stride=None,
                 rate=(1, 1),
                 padding='valid',
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if len(size) != 2:
            raise ValueError(f'`size` must be an array-like of size 2. Got: {size}')
        if stride is not None and len(stride) != 2:
            raise ValueError(f'`stride` must be an array-like of size 2. Got: {stride}')
        if rate is not None and len(rate) != 2:
            raise ValueError(f'`rate` must be an array-like of size 2. Got: {rate}')

        self.size = [1, *size, 1]
        self.stride = self.size if stride is None else [1, *stride, 1]
        self.rate = [1, 1, 1, 1] if rate is None else [1, *rate, 1]
        self.padding = padding
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        patches = tf.image.extract_patches(images=inputs, sizes=self.size, strides=self.stride, rates=self.rate,
                                           padding=self.padding.upper())
        return tf.reshape(patches, shape=[-1, patches.shape[1] * patches.shape[2], patches.shape[3]])

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size[1:-1],
            'stride': self.stride[1:-1],
            'rate': self.rate[1:-1],
            'padding': self.padding
        })
        return config
