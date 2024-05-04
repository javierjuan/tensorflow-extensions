import keras


@keras.saving.register_keras_serializable(package='tfe.layers')
class Residual2D(keras.layers.Layer):
    def __init__(self,
                 layers,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.supports_masking = True

        self._layers = layers if isinstance(layers, (list, tuple)) else [layers]
        self._convolutions = None
        self._add = keras.layers.Add()

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f'Input shape must have three dimensions. Got {input_shape}')
        output_shape = self.compute_output_shape(input_shape=input_shape)
        if len(output_shape) != 4:
            raise ValueError(f'Input `layer` does not produce a tensor shape of three dimensions. Got {output_shape}')
        if input_shape[-1] != output_shape[-1]:
            self._convolutions = []
            for layer in self._layers:
                self._convolutions.append(keras.layers.Convolution2D(
                    filters=output_shape[-1], kernel_size=(1, 1), activation=None, strides=layer.strides,
                    padding=layer.padding, data_format=layer.data_format, dilation_rate=layer.dilation_rate))
        super().build(input_shape=input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        for layer in self._layers:
            x = layer(x, training=training)
        if self._convolutions is not None:
            for layer in self._convolutions:
                inputs = layer(inputs)
        return self._add([x, inputs])

    def compute_output_shape(self, input_shape):
        for layer in self._layers:
            input_shape = layer.compute_output_shape(input_shape=input_shape)
        return input_shape

    @classmethod
    def from_config(cls, config):
        config['layers'] = [keras.layers.deserialize(layer) for layer in config['layers']]
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config.update({
            'layers': [keras.layers.serialize(layer) for layer in self._layers]
        })
        return config
