import tensorflow as tf
from keras_core import ops

from tensorflow_extensions import layers


def test_convolutional_attention():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.ConvolutionalAttention2D()
    result = layer(x)
    assert ops.all(x.shape == result.shape)
    config = layer.get_config()
    layer = layers.ConvolutionalAttention2D.from_config(config)
    assert isinstance(layer, layers.ConvolutionalAttention2D)


def test_channel_attention():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.ChannelAttention2D()
    result = layer(x)
    assert ops.all(x.shape == result.shape)
    config = layer.get_config()
    layer = layers.ChannelAttention2D.from_config(config)
    assert isinstance(layer, layers.ChannelAttention2D)


def test_spatial_attention():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.SpatialAttention2D()
    result = layer(x)
    assert ops.all(x.shape == result.shape)
    config = layer.get_config()
    layer = layers.SpatialAttention2D.from_config(config)
    assert isinstance(layer, layers.SpatialAttention2D)


def test_convolution_block():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.ConvolutionBlock2D(filters=64)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 64]))
    config = layer.get_config()
    layer = layers.ConvolutionBlock2D.from_config(config)
    assert isinstance(layer, layers.ConvolutionBlock2D)


def test_dense_block():
    x = tf.random.uniform(shape=[32, 512, 16], minval=0, maxval=1, seed=0)
    layer = layers.DenseBlock(units=64)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 512, 64]))
    config = layer.get_config()
    layer = layers.DenseBlock.from_config(config)
    assert isinstance(layer, layers.DenseBlock)


def test_fixed_embedding():
    layer = layers.FixedEmbedding(input_dim=128, output_dim=2048)
    embedding = layer(batch_size=None)
    assert ops.all(embedding.shape == tf.TensorShape([128, 2048]))
    embedding = layer(batch_size=32)
    assert ops.all(embedding.shape == tf.TensorShape([32, 128, 2048]))
    config = layer.get_config()
    layer = layers.FixedEmbedding.from_config(config)
    assert isinstance(layer, layers.FixedEmbedding)


def test_positional_encoding_1d():
    x = tf.random.uniform(shape=[32, 200, 1024], minval=0, maxval=1, seed=0)
    layer = layers.PositionalEncoding1D()
    result = layer(x)
    assert ops.all(x.shape == result.shape)
    config = layer.get_config()
    layer = layers.PositionalEncoding1D.from_config(config)
    assert isinstance(layer, layers.PositionalEncoding1D)


def test_positional_embedding_1d():
    x = tf.random.uniform(shape=[32, 200, 1024], minval=0, maxval=1, seed=0)
    layer = layers.PositionalEmbedding1D()
    result = layer(x)
    assert ops.all(x.shape == result.shape)
    config = layer.get_config()
    layer = layers.PositionalEmbedding1D.from_config(config)
    assert isinstance(layer, layers.PositionalEmbedding1D)


def test_positional_embedding_2d():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.PositionalEmbedding2D()
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224 * 224, 16]))
    config = layer.get_config()
    layer = layers.PositionalEmbedding2D.from_config(config)
    assert isinstance(layer, layers.PositionalEmbedding2D)


def test_token_position_encoding():
    x = tf.random.uniform(shape=[32, 200], minval=0, maxval=2048, seed=0, dtype=tf.int32)
    layer = layers.TokenAndPositionEncoding(vocabulary_size=2048, embedding_dimension=512)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 200, 512]))
    config = layer.get_config()
    layer = layers.TokenAndPositionEncoding.from_config(config)
    assert isinstance(layer, layers.TokenAndPositionEncoding)


def test_token_position_embedding():
    x = tf.random.uniform(shape=[32, 200], minval=0, maxval=2048, seed=0, dtype=tf.int32)
    layer = layers.TokenAndPositionEmbedding(vocabulary_size=2048, embedding_dimension=512)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 200, 512]))
    config = layer.get_config()
    layer = layers.TokenAndPositionEmbedding.from_config(config)
    assert isinstance(layer, layers.TokenAndPositionEmbedding)


def test_patch_embedding_2d():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.PatchEmbedding2D(mode='convolution', size=(8, 8), embedding_dimension=64)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224 // 8 * 224 // 8, 64]))
    config = layer.get_config()
    layer = layers.PatchEmbedding2D.from_config(config)
    assert isinstance(layer, layers.PatchEmbedding2D)
    layer = layers.PatchEmbedding2D(mode='patch', size=(8, 8), embedding_dimension=64)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224 // 8 * 224 // 8, 64]))
    config = layer.get_config()
    layer = layers.PatchEmbedding2D.from_config(config)
    assert isinstance(layer, layers.PatchEmbedding2D)


def test_inception_block():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.InceptionBlock2D(filters=64)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 64]))
    config = layer.get_config()
    layer = layers.InceptionBlock2D.from_config(config)
    assert isinstance(layer, layers.InceptionBlock2D)


def test_expand_dimensions():
    x = tf.random.uniform(shape=[32, 224, 224], minval=0, maxval=1, seed=0)
    layer = layers.ExpandDimensions(axis=-1)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 1]))
    config = layer.get_config()
    layer = layers.ExpandDimensions.from_config(config)
    assert isinstance(layer, layers.ExpandDimensions)
    layer = layers.ExpandDimensions(axis=2)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224, 1, 224]))
    config = layer.get_config()
    layer = layers.ExpandDimensions.from_config(config)
    assert isinstance(layer, layers.ExpandDimensions)


def test_repeat():
    x = tf.random.uniform(shape=[32, 224, 224], minval=0, maxval=1, seed=0)
    layer = layers.Repeat(repeats=16, axis=1)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 16, 224, 224]))
    config = layer.get_config()
    layer = layers.Repeat.from_config(config)
    assert isinstance(layer, layers.Repeat)


def test_cartesian_concatenation():
    x = tf.random.uniform(shape=[32, 100, 16], minval=0, maxval=1, seed=0)
    y = tf.random.uniform(shape=[32, 200, 16], minval=0, maxval=1, seed=0)
    layer = layers.CartesianConcatenation2D()
    result = layer([x, y])
    assert ops.all(result.shape == tf.TensorShape([32, 100, 200, 32]))
    config = layer.get_config()
    layer = layers.CartesianConcatenation2D.from_config(config)
    assert isinstance(layer, layers.CartesianConcatenation2D)


def test_math_reduce():
    x = tf.random.uniform(shape=[32, 100, 16], minval=0, maxval=1, seed=0)
    layer = layers.MathReduce(reduce_mode='max', axis=-1)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 100]))
    config = layer.get_config()
    layer = layers.MathReduce.from_config(config)
    assert isinstance(layer, layers.MathReduce)


def test_mlp():
    x = tf.random.uniform(shape=[32, 512, 16], minval=0, maxval=1, seed=0)
    layer = layers.MultiLayerPerceptron(units=64)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 512, 64]))
    config = layer.get_config()
    layer = layers.MultiLayerPerceptron.from_config(config)
    assert isinstance(layer, layers.MultiLayerPerceptron)


# def test_non_local_block():
#     x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
#     layer = layers.NonLocalBlock2D(mode='gaussian')
#     result = layer(x)
#     assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 16]))
#     config = layer.get_config()
#     layer = layers.NonLocalBlock2D.from_config(config)
#     assert isinstance(layer, layers.NonLocalBlock2D)
#     layer = layers.NonLocalBlock2D(mode='embedding')
#     result = layer(x)
#     assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 16]))
#     config = layer.get_config()
#     layer = layers.NonLocalBlock2D.from_config(config)
#     assert isinstance(layer, layers.NonLocalBlock2D)
#     layer = layers.NonLocalBlock2D(mode='concatenate')
#     result = layer(x)
#     assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 16]))
#     config = layer.get_config()
#     layer = layers.NonLocalBlock2D.from_config(config)
#     assert isinstance(layer, layers.NonLocalBlock2D)
#     layer = layers.NonLocalBlock2D(mode='dot')
#     result = layer(x)
#     assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 16]))
#     config = layer.get_config()
#     layer = layers.NonLocalBlock2D.from_config(config)
#     assert isinstance(layer, layers.NonLocalBlock2D)


def test_patch_extractor():
    x = tf.random.uniform(shape=[32, 224, 224, 3], minval=0, maxval=1, seed=0)
    layer = layers.PatchExtractor2D(size=(8, 8))
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224 // 8, 224 // 8, 8 * 8 * 3]))
    config = layer.get_config()
    layer = layers.PatchExtractor2D.from_config(config)
    assert isinstance(layer, layers.PatchExtractor2D)


def test_channel_average_pooling():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.ChannelAveragePooling()
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224, 224]))
    config = layer.get_config()
    layer = layers.ChannelAveragePooling.from_config(config)
    assert isinstance(layer, layers.ChannelAveragePooling)


def test_channel_max_pooling():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.ChannelMaxPooling()
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224, 224]))
    config = layer.get_config()
    layer = layers.ChannelMaxPooling.from_config(config)
    assert isinstance(layer, layers.ChannelMaxPooling)


def test_residual_block():
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.ResidualBlock2D(layer=layers.ConvolutionBlock2D(filters=64))
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 64]))
    config = layer.get_config()
    layer = layers.ResidualBlock2D.from_config(config)
    assert isinstance(layer, layers.ResidualBlock2D)


def test_transformer():
    x = tf.random.uniform(shape=[32, 200, 1024], minval=0, maxval=1, seed=0)
    y = tf.random.uniform(shape=[32, 150, 1024], minval=0, maxval=1, seed=0)
    layer = layers.Transformer(encoder_units=512, encoder_num_heads=8)
    result = layer([x, y])
    assert ops.all(result.shape == tf.TensorShape([32, 150, 1024]))
    config = layer.get_config()
    layer = layers.Transformer.from_config(config)
    assert isinstance(layer, layers.Transformer)
