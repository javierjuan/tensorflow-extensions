import tensorflow as tf


class Standard2D(tf.keras.layers.Layer):
    def __init__(self,
                 height,
                 width,
                 crop_to_aspect_ratio=False,
                 mean=None,
                 variance=None,
                 invert=False,
                 interpolation='bilinear',
                 axis=-1,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.height = height
        self.width = width
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.mean = mean
        self.variance = variance
        self.invert = invert
        self.interpolation = interpolation
        self.axis = axis

        self.resizing = tf.keras.layers.Resizing(
            height=height, width=width, interpolation=interpolation, crop_to_aspect_ratio=crop_to_aspect_ratio)
        self.normalization = tf.keras.layers.Normalization(axis=axis, mean=mean, variance=variance, invert=invert)

    def adapt(self, data, batch_size=None, steps=None):
        self.normalization.adapt(data=data, batch_size=batch_size, steps=steps)

    def call(self, inputs, **kwargs):
        x = self.resizing(inputs)
        x = self.normalization(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'height': self.height,
            'width': self.width,
            'crop_to_aspect_ratio': self.crop_to_aspect_ratio,
            'mean': self.mean,
            'variance': self.variance,
            'invert': self.invert,
            'interpolation': self.interpolation,
            'axis': self.axis
        })
