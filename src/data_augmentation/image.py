import tensorflow as tf
import keras_cv as kcv


class RandomAugmentation(tf.keras.layers.Layer):
    def __init__(self,
                 augmentations_per_image=3,
                 magnitude=0.5,
                 mode='horizontal_and_vertical',
                 value_range=(0, 255),
                 crop_area_factor=(0.8, 1.0),
                 aspect_ratio_factor=(0.75, 0.75),
                 rate=0.5,
                 cut_mix_alpha=1.0,
                 mix_up_alpha=0.3,
                 interpolation='bilinear',
                 seed=None,
                 bounding_box_format=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.augmentations_per_image = augmentations_per_image
        self.magnitude = magnitude
        self.mode = mode
        self.value_range = value_range
        self.crop_area_factor = crop_area_factor
        self.aspect_ratio_factor = aspect_ratio_factor
        self.rate = rate
        self.cut_mix_alpha = cut_mix_alpha
        self.mix_up_alpha = mix_up_alpha
        self.interpolation = interpolation
        self.seed = seed
        self.bounding_box_format = bounding_box_format

        self.random_flip = kcv.layers.RandomFlip(
            mode=mode, rate=rate, seed=seed, bounding_box_format=bounding_box_format)
        self.crop_and_resize = None
        self.random_augment = kcv.layers.RandAugment(
            augmentations_per_image=augmentations_per_image, magnitude=magnitude, value_range=value_range)
        self.cut_mix_or_mix_up = kcv.layers.RandomChoice(
            layers=[kcv.layers.CutMix(alpha=cut_mix_alpha, seed=seed), kcv.layers.MixUp(alpha=mix_up_alpha, seed=seed)],
            batchwise=True)

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.crop_and_resize = kcv.layers.RandomCropAndResize(
            target_size=input_shape[1:-1], crop_area_factor=self.crop_area_factor,
            aspect_ratio_factor=self.aspect_ratio_factor, interpolation=self.interpolation,
            bounding_box_format=self.bounding_box_format, seed=self.seed)

    def call(self, inputs, training=False, **kwargs):
        x = self.random_flip(inputs, training=training)
        x = self.crop_and_resize(x, training=training)
        x = self.random_augment(x, training=training)
        x = self.cut_mix_or_mix_up(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'augmentations_per_image': self.augmentations_per_image,
            'magnitude': self.magnitude,
            'mode': self.mode,
            'value_range': self.value_range,
            'crop_area_factor': self.crop_area_factor,
            'aspect_ratio_factor': self.aspect_ratio_factor,
            'rate': self.rate,
            'cut_mix_alpha': self.cut_mix_alpha,
            'mix_up_alpha': self.mix_up_alpha,
            'interpolation': self.interpolation,
            'seed': self.seed,
            'bounding_box_format': self.bounding_box_format
        })
