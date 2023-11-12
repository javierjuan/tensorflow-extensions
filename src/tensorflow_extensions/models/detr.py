import tensorflow as tf

from ..layers.embedding import FixedEmbedding
from ..layers.encoding import PositionalEmbedding2D
from ..layers.mlp import MultiLayerPerceptron
from ..layers.transformer import Transformer
from ..metrics import Hungarian
from ..models.model import Model


@tf.keras.saving.register_keras_serializable(package='tfe.models')
class DETRModel(Model):
    def __init__(self, padding_axis=-1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.metric = Hungarian(padding_axis=padding_axis, name='hungarian')

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        del x
        loss = self.loss(y_true=y, y_pred=y_pred, sample_weight=sample_weight)
        if self.losses:
            loss += tf.math.add_n(self.losses)
        self.loss_tracker.update_state(loss)
        return loss

    def compute_metrics(self, x, y, y_pred, sample_weight):
        self.metric.update_state(y_true=y, y_pred=y_pred)
        return super().compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)


@tf.keras.saving.register_keras_serializable(package='tfe.models')
class DETR(DETRModel):
    def __init__(self,
                 num_queries,
                 num_labels,
                 embedding_dimension,
                 encoder_units,
                 encoder_num_heads,
                 backbone=None,
                 train_backbone=True,
                 decoder_units=None,
                 decoder_num_heads=None,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activation='mish',
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 input_length=None,
                 sparse=False,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 convolution_groups=1,
                 rate=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        encoder_units = encoder_units if isinstance(encoder_units, (tuple, list)) else [encoder_units]

        self.num_queries = num_queries
        self.num_labels = num_labels
        self.embedding_dimension = embedding_dimension
        self.encoder_units = encoder_units
        self.encoder_num_heads = encoder_num_heads
        self.train_backbone = train_backbone
        self.decoder_units = decoder_units
        self.decoder_num_heads = decoder_num_heads
        self.use_bias = use_bias
        self._output_shape = output_shape
        self.attention_axes = attention_axes
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.activation = activation
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.input_length = input_length
        self.sparse = sparse
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.convolution_groups = convolution_groups
        self.rate = rate
        self.seed = seed

        self.backbone = tf.keras.applications.resnet.ResNet50(include_top=False) if backbone is None else backbone
        self.convolution = tf.keras.layers.Convolution2D(
            filters=embedding_dimension, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, groups=convolution_groups, activation=None, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.positional_embedding = PositionalEmbedding2D(
            embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint)
        self.transformer = Transformer(
            encoder_units=encoder_units, encoder_num_heads=encoder_num_heads, decoder_units=decoder_units,
            decoder_num_heads=decoder_num_heads, use_bias=use_bias, output_shape=output_shape,
            attention_axes=attention_axes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, activation=activation, axis=axis, epsilon=epsilon, center=center,
            scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, rate=rate, seed=seed)
        self.label = MultiLayerPerceptron(
            units=[128, num_labels + 1], activation=['mish', 'softmax'], use_bias=use_bias,
            normalization=['layer', None], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.bounding_box = MultiLayerPerceptron(
            units=[64, 16, 4], activation=['mish', 'mish', 'sigmoid'], use_bias=use_bias,
            normalization=['layer', 'layer', None], kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)
        self.query = FixedEmbedding(
            input_dim=self.num_queries, output_dim=self.encoder_units[-1],
            embeddings_initializer=self.embeddings_initializer, embeddings_regularizer=self.embeddings_regularizer,
            embeddings_constraint=self.embeddings_constraint)

    def build(self, input_shape):
        if not isinstance(self.backbone, tf.keras.Model):
            raise TypeError('`backbone` must be an instance of Keras Model')

        if not self.train_backbone:
            for layer in self.backbone.layers:
                layer.trainable = False
        super().build(input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs, training=training)
        x = self.convolution(x)
        x = self.positional_embedding(x)
        x = self.transformer([x, self.query(batch_size=tf.shape(inputs)[0])], training=training)
        return {'label': self.label(x), 'bounding_box': self.bounding_box(x)}

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_queries': self.num_queries,
            'num_labels': self.num_labels,
            'embedding_dimension': self.embedding_dimension,
            'encoder_units': self.encoder_units,
            'encoder_num_heads': self.encoder_num_heads,
            'backbone': self.backbone,
            'train_backbone': self.train_backbone,
            'decoder_units': self.decoder_units,
            'decoder_num_heads': self.decoder_num_heads,
            'use_bias': self.use_bias,
            'output_shape': self._output_shape,
            'attention_axes': self.attention_axes,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'activation': self.activation,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'input_length': self.input_length,
            'sparse': self.sparse,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'convolution_groups': self.convolution_groups,
            'rate': self.rate,
            'seed': self.seed
        })
        return config
