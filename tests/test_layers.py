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
    x = tf.random.uniform(shape=[32, 224, 224, 16], minval=0, maxval=1, seed=0)
    layer = layers.DenseBlock(units=64)
    result = layer(x)
    assert ops.all(result.shape == tf.TensorShape([32, 224, 224, 64]))
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
