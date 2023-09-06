import tensorflow as tf


class ImageRGB(tf.keras.layers.Layer):
    def __init__(self,
                 flip_mode='horizontal_and_vertical',
                 rotation_factor=0.5,
                 zoom_factor=0.5,
                 brightness_factor=0.5,
                 contrast_factor=0.5,
                 value_range=(0, 255),
                 fill_mode='reflect',
                 interpolation='bilinear',
                 seed=None,
                 fill_value=0.0,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.flip_mode = flip_mode
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.value_range = value_range
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.seed = seed
        self.fill_value = fill_value

        self.flip = tf.keras.layers.RandomFlip(mode=flip_mode, seed=seed)
        self.rotation = tf.keras.layers.RandomRotation(
            factor=rotation_factor, fill_mode=fill_mode, interpolation=interpolation, seed=seed, fill_value=fill_value)
        self.zoom = tf.keras.layers.RandomZoom(
            height_factor=zoom_factor, width_factor=None, fill_mode=fill_mode, interpolation=interpolation, seed=seed,
            fill_value=fill_value)
        self.brightness = tf.keras.layers.RandomBrightness(factor=brightness_factor, value_range=value_range, seed=seed)
        self.contrast = tf.keras.layers.RandomContrast(factor=contrast_factor, seed=seed)

    def call(self, inputs, training=False, **kwargs):
        x = self.flip(inputs, training=training)
        x = self.rotation(x, training=training)
        x = self.zoom(x, training=training)
        x = self.brightness(x, training=training)
        x = self.contrast(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'flip_mode': self.flip_mode,
            'rotation_factor': self.rotation_factor,
            'zoom_factor': self.zoom_factor,
            'brightness_factor': self.brightness_factor,
            'contrast_factor': self.contrast_factor,
            'value_range': self.value_range,
            'fill_mode': self.fill_mode,
            'interpolation': self.interpolation,
            'seed': self.seed,
            'fill_value': self.fill_value
        })
