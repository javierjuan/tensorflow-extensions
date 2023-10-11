import tensorflow as tf


@tf.function
def _label_smoothing(y_true, label_smoothing):
    num_labels = tf.cast(tf.shape(y_true)[-1], tf.float32)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_labels)


@tf.function
def _initialize_loss(y_true, y_pred, label_smoothing=0.1, from_logits=False):
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), dtype=tf.float32)
    y_true = tf.cast(tf.convert_to_tensor(y_true), dtype=tf.float32)
    label_smoothing = tf.cast(tf.convert_to_tensor(label_smoothing), dtype=tf.float32)

    if not y_true.shape.is_compatible_with(y_pred.shape):
        raise ValueError(f'Incompatible shapes for y_true: {y_true.shape} and y_pred: {y_pred.shape}')

    y_pred = tf.math.softmax(logits=y_pred) if from_logits else y_pred

    # Perform ALWAYS a small label smoothing to prevent division by zero in metrics. This is better to add an epsilon
    # value at the numerator and denominator of each metric
    label_smoothing = 0.1 if label_smoothing < 0.1 else label_smoothing
    y_true = _label_smoothing(y_true=y_true, label_smoothing=label_smoothing)

    return y_true, y_pred


@tf.function
def _finalize_loss(loss, reduction='sum', label_penalties=None):
    if label_penalties is not None:
        label_penalties = 1.0 + tf.cast(tf.convert_to_tensor(label_penalties), dtype=tf.float32)
        if not label_penalties.shape.is_compatible_with(loss.shape):
            label_penalties = tf.broadcast_to(label_penalties, loss.shape)
        loss = tf.multiply(loss, label_penalties)

    if reduction == 'sum':
        loss = tf.reduce_sum(loss, axis=-1)
    else:
        loss = tf.reduce_mean(loss, axis=-1)

    return loss


@tf.function
def _dice_score(y_true, y_pred):
    # IMPORTANT: The denominator MUST be squared for mathematical correctness. Dice metric is defined for discrete
    # sets of labels. For continuous probability arrays the denominator must be squared in order to accomplish with
    # the concept of `cardinality` -> |A| = sum(a_i^2).
    axis_reduce = tf.range(start=1, limit=tf.rank(y_pred) - 1)
    numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis_reduce)
    denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=axis_reduce)
    return tf.math.divide_no_nan(numerator, denominator)


@tf.function
def _dice_loss(y_true, y_pred, label_smoothing=0.1, label_penalties=None, from_logits=False):
    y_true, y_pred = _initialize_loss(y_true=y_true, y_pred=y_pred, label_smoothing=label_smoothing,
                                      from_logits=from_logits)
    loss = 1.0 - _dice_score(y_true=y_true, y_pred=y_pred)
    return _finalize_loss(loss=loss, label_penalties=label_penalties)


class Dice(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, label_penalties=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='dice'):
        super().__init__(name=name, reduction=reduction)

        self.label_smoothing = label_smoothing
        self.label_penalties = label_penalties
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return _dice_loss(y_true, y_pred, label_smoothing=self.label_smoothing,
                          label_penalties=self.label_penalties, from_logits=self.from_logits)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'label_penalties': self.label_penalties,
            'from_logits': self.from_logits
        })
        return config


@tf.function
def _jaccard_loss(y_true, y_pred, label_smoothing=0.1, label_penalties=None, from_logits=False):
    y_true, y_pred = _initialize_loss(y_true=y_true, y_pred=y_pred, label_smoothing=label_smoothing,
                                      from_logits=from_logits)
    scores = _dice_score(y_true=y_true, y_pred=y_pred)
    loss = 1.0 - (scores / (2.0 - scores))
    return _finalize_loss(loss=loss, label_penalties=label_penalties)


class Jaccard(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, label_penalties=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='jaccard'):
        super().__init__(name=name, reduction=reduction)

        self.label_smoothing = label_smoothing
        self.label_penalties = label_penalties
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return _jaccard_loss(y_true, y_pred, label_smoothing=self.label_smoothing,
                             label_penalties=self.label_penalties, from_logits=self.from_logits)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'label_penalties': self.label_penalties,
            'from_logits': self.from_logits
        })
        return config


@tf.function
def _cross_entropy_loss(y_true, y_pred, label_smoothing=0.1, from_logits=False):
    loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=from_logits,
                                                    label_smoothing=label_smoothing)
    # Reduce to be compatible with the other losses
    axis_reduce = tf.range(start=1, limit=tf.rank(loss))
    return tf.reduce_mean(loss, axis=axis_reduce)


@tf.function
def _dice_plus_cross_entropy_loss(y_true, y_pred, label_smoothing=0.1, label_penalties=None, from_logits=False):
    loss_cross_entropy = _cross_entropy_loss(y_true=y_true, y_pred=y_pred, label_smoothing=label_smoothing,
                                             from_logits=from_logits)
    loss_dice = _dice_loss(y_true, y_pred, label_smoothing=label_smoothing, label_penalties=label_penalties,
                           from_logits=from_logits)
    return loss_cross_entropy + loss_dice


class DicePlusCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, label_penalties=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='dice_plus_categorical_cross_entropy'):
        super().__init__(name=name, reduction=reduction)

        self.label_smoothing = label_smoothing
        self.label_penalties = label_penalties
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return _dice_plus_cross_entropy_loss(y_true, y_pred, label_smoothing=self.label_smoothing,
                                             label_penalties=self.label_penalties, from_logits=self.from_logits)

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'label_penalties': self.label_penalties,
            'from_logits': self.from_logits
        })
        return config


@tf.function
def _tversky_score(y_true, y_pred, alpha=0.5, beta=0.5):
    # WARNING!: This is not a mathematically correct formula. p1 = 1 - y_pred and g1 = 1 - y_true try to imitate the
    # operation `relative complement` (|A\B|) of two discrete sets of labels, but p1 and g1 are not equivalents of
    # this operation. Therefore, the metric does not behave correctly for perfectly matching target and labels

    p0 = y_pred
    p1 = 1 - y_pred
    g0 = y_true
    g1 = 1 - y_true

    tp = p0 * g0
    fp = alpha * p0 * g1
    fn = beta * p1 * g0

    axis_reduce = tf.range(1, tf.rank(y_pred) - 1)
    numerator = tf.math.reduce_sum(tp, axis=axis_reduce)
    denominator = tf.math.reduce_sum(tp + tf.math.square(fp) + tf.math.square(fn), axis=axis_reduce)
    return tf.math.divide_no_nan(numerator, denominator)


@tf.function
def _tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, label_smoothing=0.1, label_penalties=None,
                  from_logits=False):
    y_true, y_pred = _initialize_loss(y_true=y_true, y_pred=y_pred, label_smoothing=label_smoothing,
                                      from_logits=from_logits)
    loss = 1.0 - _tversky_score(y_true=y_true, y_pred=y_pred, alpha=alpha, beta=beta, )
    return _finalize_loss(loss=loss, label_penalties=label_penalties)


class Tversky(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.5, label_smoothing=0.1, label_penalties=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='tversky'):
        super().__init__(name=name, reduction=reduction)

        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.label_penalties = label_penalties
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return _tversky_loss(y_true, y_pred, alpha=self.alpha, beta=self.beta,
                             label_smoothing=self.label_smoothing, label_penalties=self.label_penalties,
                             from_logits=self.from_logits)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'label_smoothing': self.label_smoothing,
            'label_penalties': self.label_penalties,
            'from_logits': self.from_logits
        })
        return config
