import keras_core as keras
from keras_core import ops
from ..utils import normalize_axis


@keras.saving.register_keras_serializable(package='tfe.layers')
class ExpandDimensions(keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return ops.expand_dims(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        axis = normalize_axis(self.axis, len(input_shape) + 1)
        return input_shape[:axis] + (1,) + input_shape[axis:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class Repeat(keras.layers.Layer):
    def __init__(self,
                 repeats,
                 axis,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.repeats = repeats
        self.axis = axis

    def call(self, inputs, **kwargs):
        return ops.repeat(ops.expand_dims(inputs, axis=self.axis), repeats=self.repeats, axis=self.axis)

    def compute_output_shape(self, input_shape):
        axis = normalize_axis(self.axis, len(input_shape) + 1)
        return input_shape[:axis] + (self.repeats,) + input_shape[axis:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'repeats': self.repeats,
            'axis': self.axis
        })
        return config


@keras.saving.register_keras_serializable(package='tfe.layers')
class CartesianConcatenation2D(keras.layers.Layer):
    def __init__(self,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        x, y = inputs
        tile_x = ops.tile(ops.expand_dims(x, axis=2), repeats=[1, 1, y.shape[1], 1])
        tile_y = ops.tile(ops.expand_dims(y, axis=1), repeats=[1, x.shape[1], 1, 1])
        return ops.concatenate([tile_x, tile_y], axis=-1)

    def compute_output_shape(self, input_shape):
        x_shape, y_shape = input_shape
        return x_shape[0], x_shape[1], y_shape[1], x_shape[2] + y_shape[2]


@keras.saving.register_keras_serializable(package='tfe.layers')
class MathReduce(keras.layers.Layer):
    def __init__(self,
                 reduce_mode,
                 axis=-1,
                 keepdims=False,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduce_mode = reduce_mode
        self.axis = axis
        self.keepdims = keepdims

        if reduce_mode == 'min':
            self.reduce_operation = ops.min
        elif reduce_mode == 'max':
            self.reduce_operation = ops.max
        elif reduce_mode == 'mean':
            self.reduce_operation = ops.mean
        elif reduce_mode == 'sum':
            self.reduce_operation = ops.sum
        elif reduce_mode == 'prod':
            self.reduce_operation = ops.prod
        else:
            raise ValueError(f'Unexpected operation {reduce_mode}')

    def call(self, inputs, **kwargs):
        return self.reduce_operation(inputs, axis=self.axis, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape):
        axis = normalize_axis(self.axis, len(input_shape))
        if self.keepdims:
            return input_shape[:axis] + (1,) + input_shape[axis + 1:]
        else:
            return input_shape[:axis] + input_shape[axis + 1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduce_mode': self.reduce_mode,
            'axis': self.axis,
            'keepdims': self.keepdims
        })
        return config
