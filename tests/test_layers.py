import inspect

import tensorflow as tf
from keras import ops

from tensorflow_extensions import layers


def _test_serialization(layer, layer_class):
    config = layer.get_config()
    init_signature = inspect.signature(layer_class.__init__)
    for parameter in init_signature.parameters.values():
        if parameter.name in ('self', 'args', 'kwargs'):
            continue
        if parameter.name not in config.keys() or parameter.name not in layer.__dict__.keys():
            raise AssertionError(f'Parameter {parameter.name} not found in `config` or in {layer_class} attributes')


def _test_generic_layer(layer, layer_class, x, input_shape, output_shape, **kwargs):
    result = layer(x, **kwargs)
    assert ops.all(output_shape == result.shape)
    assert ops.all(output_shape == layer.compute_output_shape(input_shape=input_shape))
    assert isinstance(layer_class.from_config(layer.get_config()), layer_class)
    _test_serialization(layer=layer, layer_class=layer_class)


def test_convolutional_attention():
    input_shape, output_shape = (32, 240, 240, 16), (32, 240, 240, 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.ConvolutionalAttention2D(), layers.ConvolutionalAttention2D, x, input_shape,
                        output_shape)


def test_channel_attention():
    input_shape, output_shape = (32, 240, 240, 16), (32, 240, 240, 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.ChannelAttention2D(), layers.ChannelAttention2D, x, input_shape, output_shape)


def test_spatial_attention():
    input_shape, output_shape = (32, 240, 240, 16), (32, 240, 240, 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.SpatialAttention2D(), layers.SpatialAttention2D, x, input_shape, output_shape)


def test_convolution_block():
    input_shape = (32, 240, 240, 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.ConvolutionBlock2D(filters=64), layers.ConvolutionBlock2D, x, input_shape,
                        (32, 240, 240, 64))
    _test_generic_layer(layers.ConvolutionBlock2D(filters=64, kernel_size=(5, 5)), layers.ConvolutionBlock2D, x,
                        input_shape,
                        (32, 240, 240, 64))
    _test_generic_layer(layers.ConvolutionBlock2D(filters=64, padding='valid'), layers.ConvolutionBlock2D, x,
                        input_shape,
                        (32, 238, 238, 64))
    _test_generic_layer(layers.ConvolutionBlock2D(filters=32, strides=(2, 2)), layers.ConvolutionBlock2D, x,
                        input_shape,
                        (32, 240 // 2, 240 // 2, 32))
    _test_generic_layer(layers.ConvolutionBlock2D(filters=32, padding='valid', strides=(2, 2)),
                        layers.ConvolutionBlock2D,
                        x,
                        input_shape, (32, 238 // 2, 238 // 2, 32))


def test_dense_block():
    input_shape, output_shape = (32, 512, 16), (32, 512, 64)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.DenseBlock(units=64), layers.DenseBlock, x, input_shape, output_shape)


def test_fixed_embedding():
    layer = layers.FixedEmbedding(input_dim=128, output_dim=2048)
    embedding = layer(batch_size=None)
    assert ops.all(embedding.shape == tf.TensorShape([128, 2048]))
    assert ops.all(tf.TensorShape((128, 2048)) == embedding.shape)
    assert ops.all(tf.TensorShape((128, 2048)) == layer.compute_output_shape(batch_size=None))
    assert isinstance(layers.FixedEmbedding.from_config(layer.get_config()), layers.FixedEmbedding)
    embedding = layer(batch_size=32)
    assert ops.all(tf.TensorShape((32, 128, 2048)) == embedding.shape)
    assert ops.all(tf.TensorShape((32, 128, 2048)) == layer.compute_output_shape(batch_size=32))
    assert isinstance(layers.FixedEmbedding.from_config(layer.get_config()), layers.FixedEmbedding)
    _test_serialization(layer=layer, layer_class=layers.FixedEmbedding)


def test_positional_encoding_1d():
    input_shape, output_shape = (32, 512, 1024), (32, 512, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.PositionalEncoding1D(), layers.PositionalEncoding1D, x, input_shape, output_shape)


def test_positional_embedding_1d():
    input_shape, output_shape = (32, 512, 1024), (32, 512, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.PositionalEmbedding1D(), layers.PositionalEmbedding1D, x, input_shape, output_shape)


def test_positional_embedding_2d():
    input_shape, output_shape = (32, 240, 240, 16), (32, 240 * 240, 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.PositionalEmbedding2D(), layers.PositionalEmbedding2D, x, input_shape, output_shape)


def test_token_position_encoding():
    input_shape, output_shape = (32, 512), (32, 512, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=2048, seed=0, dtype=tf.int32)
    _test_generic_layer(layers.TokenAndPositionEncoding(vocabulary_size=2048, embedding_dimension=1024),
                        layers.TokenAndPositionEncoding, x, input_shape, output_shape)


def test_token_position_embedding():
    input_shape, output_shape = (32, 512), (32, 512, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=2048, seed=0, dtype=tf.int32)
    _test_generic_layer(layers.TokenAndPositionEmbedding(vocabulary_size=2048, embedding_dimension=1024),
                        layers.TokenAndPositionEmbedding, x, input_shape, output_shape)


def test_patch_embedding_2d():
    input_shape, output_shape = (32, 240, 240, 16), (32, 240 // 8, 240 // 8, 64)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.PatchEmbedding2D(mode='convolution', size=(8, 8), embedding_dimension=64),
                        layers.PatchEmbedding2D, x, input_shape, output_shape)
    _test_generic_layer(layers.PatchEmbedding2D(mode='patch', size=(8, 8), embedding_dimension=64),
                        layers.PatchEmbedding2D, x, input_shape, output_shape)


def test_inception_block():
    input_shape = (32, 240, 240, 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.InceptionBlock2D(filters=64), layers.InceptionBlock2D, x, input_shape,
                        (32, 240, 240, 64))
    _test_generic_layer(layers.InceptionBlock2D(filters=64, strides=(2, 2)), layers.InceptionBlock2D, x,
                        input_shape, (32, 240 // 2, 240 // 2, 64))


def test_expand_dimensions():
    input_shape = (32, 240, 240)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.ExpandDimensions(axis=-1), layers.ExpandDimensions, x, input_shape, (32, 240, 240, 1))
    _test_generic_layer(layers.ExpandDimensions(axis=2), layers.ExpandDimensions, x, input_shape, (32, 240, 1, 240))


def test_repeat():
    input_shape = (32, 240, 240)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.Repeat(repeats=16, axis=-1), layers.Repeat, x, input_shape, (32, 240, 240, 16))
    _test_generic_layer(layers.Repeat(repeats=16, axis=1), layers.Repeat, x, input_shape, (32, 16, 240, 240))


def test_cartesian_concatenation():
    x = tf.random.uniform(shape=(32, 240, 16), minval=0, maxval=1, seed=0)
    y = tf.random.uniform(shape=(32, 480, 32), minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.CartesianConcatenation2D(), layers.CartesianConcatenation2D, x,
                        [(32, 240, 16), (32, 480, 32)], (32, 240, 480, 48), y=y)


def test_math_reduce():
    input_shape = (32, 240, 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.MathReduce(reduce_mode='max', axis=1), layers.MathReduce, x, input_shape, (32, 16))
    _test_generic_layer(layers.MathReduce(reduce_mode='mean', axis=-1), layers.MathReduce, x, input_shape, (32, 240))


def test_mlp():
    input_shape, output_shape = (32, 512, 16), (32, 512, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.MultiLayerPerceptron(units=1024), layers.MultiLayerPerceptron, x, input_shape,
                        output_shape)


# def test_non_local_block():
#     x = tf.random.uniform(shape=[32, 240, 240, 16], minval=0, maxval=1, seed=0)
#     layer = layers.NonLocalBlock2D(mode='gaussian')
#     result = layer(x)
#     assert ops.all(result.shape == tf.TensorShape([32, 240, 240, 16]))
#     config = layer.get_config()
#     layer = layers.NonLocalBlock2D.from_config(config)
#     assert isinstance(layer, layers.NonLocalBlock2D)
#     layer = layers.NonLocalBlock2D(mode='embedding')
#     result = layer(x)
#     assert ops.all(result.shape == tf.TensorShape([32, 240, 240, 16]))
#     config = layer.get_config()
#     layer = layers.NonLocalBlock2D.from_config(config)
#     assert isinstance(layer, layers.NonLocalBlock2D)
#     layer = layers.NonLocalBlock2D(mode='concatenate')
#     result = layer(x)
#     assert ops.all(result.shape == tf.TensorShape([32, 240, 240, 16]))
#     config = layer.get_config()
#     layer = layers.NonLocalBlock2D.from_config(config)
#     assert isinstance(layer, layers.NonLocalBlock2D)
#     layer = layers.NonLocalBlock2D(mode='dot')
#     result = layer(x)
#     assert ops.all(result.shape == tf.TensorShape([32, 240, 240, 16]))
#     config = layer.get_config()
#     layer = layers.NonLocalBlock2D.from_config(config)
#     assert isinstance(layer, layers.NonLocalBlock2D)


def test_patch_extractor():
    input_shape, output_shape = (32, 240, 240, 16), (32, 240 // 8, 240 // 8, 8 * 8 * 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.PatchExtractor2D(size=(8, 8)), layers.PatchExtractor2D, x, input_shape, output_shape)


def test_channel_average_pooling():
    input_shape, output_shape = (32, 240, 240, 16), (32, 240, 240)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.ChannelAveragePooling(), layers.ChannelAveragePooling, x, input_shape, output_shape)


def test_channel_max_pooling():
    input_shape, output_shape = (32, 240, 240, 16), (32, 240, 240)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.ChannelMaxPooling(), layers.ChannelMaxPooling, x, input_shape, output_shape)


def test_residual_block():
    input_shape = (32, 240, 240, 16)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.ResidualBlock2D(layer=layers.ConvolutionBlock2D(filters=64)), layers.ResidualBlock2D, x,
                        input_shape, (32, 240, 240, 64))
    _test_generic_layer(layers.ResidualBlock2D(layer=layers.ConvolutionBlock2D(filters=64, kernel_size=(5, 5))),
                        layers.ResidualBlock2D, x, input_shape, (32, 240, 240, 64))
    _test_generic_layer(layers.ResidualBlock2D(layer=layers.ConvolutionBlock2D(filters=64, kernel_size=(5, 5),
                                                                               strides=(4, 4))), layers.ResidualBlock2D,
                        x,
                        input_shape, (32, 240 // 4, 240 // 4, 64))


def test_transformer_encoder():
    input_shape = (32, 512, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.TransformerEncoder(units=[512, 256], num_heads=8), layers.TransformerEncoder,
                        x, input_shape, input_shape)


def test_transformer_encoder_layer():
    input_shape = (32, 512, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    _test_generic_layer(layers.TransformerEncoderLayer(units=512, num_heads=8), layers.TransformerEncoderLayer,
                        x, input_shape, input_shape)


def test_transformer_decoder():
    context_shape, input_shape = (32, 512, 1024), (32, 256, 1024)
    x = tf.random.uniform(shape=context_shape, minval=0, maxval=1, seed=0)
    y = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    layer = layers.TransformerDecoder(units=[512, 256], num_heads=8)
    result = layer(y, context=None)
    assert ops.all(input_shape == result.shape)
    assert ops.all(input_shape == layer.compute_output_shape(input_shape=input_shape, context_shape=None))
    assert isinstance(layers.TransformerDecoder.from_config(layer.get_config()), layers.TransformerDecoder)
    _test_serialization(layer=layer, layer_class=layers.TransformerDecoder)
    result = layer(y, context=x)
    assert ops.all(input_shape == result.shape)
    assert ops.all(input_shape == layer.compute_output_shape(input_shape=input_shape, context_shape=context_shape))
    assert isinstance(layers.TransformerDecoder.from_config(layer.get_config()), layers.TransformerDecoder)
    _test_serialization(layer=layer, layer_class=layers.TransformerDecoder)


def test_transformer_decoder_layer():
    context_shape, input_shape = (32, 512, 1024), (32, 256, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    y = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    layer = layers.TransformerDecoderLayer(units=512, num_heads=8)
    result = layer(y, context=None)
    assert ops.all(input_shape == result.shape)
    assert ops.all(input_shape == layer.compute_output_shape(input_shape=input_shape, context_shape=None))
    assert isinstance(layers.TransformerDecoderLayer.from_config(layer.get_config()), layers.TransformerDecoderLayer)
    _test_serialization(layer=layer, layer_class=layers.TransformerDecoderLayer)
    result = layer(y, context=x)
    assert ops.all(input_shape == result.shape)
    assert ops.all(input_shape == layer.compute_output_shape(input_shape=input_shape, context_shape=context_shape))
    assert isinstance(layers.TransformerDecoderLayer.from_config(layer.get_config()), layers.TransformerDecoderLayer)
    _test_serialization(layer=layer, layer_class=layers.TransformerDecoderLayer)


def test_transformer():
    input_shape, output_shape = (32, 512, 1024), (32, 256, 1024)
    x = tf.random.uniform(shape=input_shape, minval=0, maxval=1, seed=0)
    y = tf.random.uniform(shape=output_shape, minval=0, maxval=1, seed=0)
    layer = layers.Transformer(encoder_units=[512, 256], encoder_num_heads=8)
    result = layer(inputs=x, outputs=y)
    assert ops.all(output_shape == result.shape)
    assert ops.all(output_shape == layer.compute_output_shape(input_shape=input_shape, output_shape=output_shape))
    assert isinstance(layers.Transformer.from_config(layer.get_config()), layers.Transformer)
    _test_serialization(layer=layer, layer_class=layers.Transformer)
