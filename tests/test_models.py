import keras
import tensorflow as tf
from keras import ops

from tensorflow_extensions import layers


def test_train_convolutional_classification_model():
    x = tf.random.uniform(shape=[256, 240, 240, 3], minval=0, maxval=1, seed=0)
    y = tf.random.uniform(shape=[256], minval=0, maxval=10, seed=0, dtype=tf.int32)
    y = ops.one_hot(y, num_classes=10)
    model = keras.Sequential([
        keras.Input(shape=(240, 240, 3)),
        layers.ResidualBlock2D(layers.ConvolutionBlock2D(filters=8, kernel_size=(3, 3), normalization='batch',
                                                         activation='mish', rate=0.1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        layers.ResidualBlock2D(layers.InceptionBlock2D(filters=16, normalization='layer', activation='relu', rate=0.2)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        layers.ResidualBlock2D(layers.ConvolutionBlock2D(filters=32, kernel_size=(3, 3), normalization='batch',
                                                         activation='swish')),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        layers.ResidualBlock2D(layers.InceptionBlock2D(filters=64, normalization='batch', activation='relu')),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        layers.MultiLayerPerceptron(units=[512, 128, 32], activation=['relu', 'mish', 'swish'],
                                    normalization='layer', rate=[0.1, 0.2, None]),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3)
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(x, y, batch_size=8)
