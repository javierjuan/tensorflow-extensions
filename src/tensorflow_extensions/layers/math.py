import tensorflow as tf


class ExpandDimensions(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis
        })
        return config


class MathReduce(tf.keras.layers.Layer):
    def __init__(self,
                 reduce_mode,
                 axis=-1,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduce_mode = reduce_mode
        self.axis = axis

        if reduce_mode == 'min':
            self.reduce_operation = tf.math.reduce_min
        elif reduce_mode == 'max':
            self.reduce_operation = tf.math.reduce_max
        elif reduce_mode == 'mean':
            self.reduce_operation = tf.math.reduce_mean
        elif reduce_mode == 'sum':
            self.reduce_operation = tf.math.reduce_sum
        elif reduce_mode == 'prod':
            self.reduce_operation = tf.math.reduce_prod
        else:
            raise ValueError(f'Unexpected operation {reduce_mode}')

    def call(self, inputs, **kwargs):
        return self.reduce_operation(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduce_mode': self.reduce_mode,
            'axis': self.axis
        })
        return config


class MatrixMultiplication(tf.keras.layers.Layer):
    def call(self, a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, **kwargs):
        return tf.linalg.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a,
                                adjoint_b=adjoint_b)