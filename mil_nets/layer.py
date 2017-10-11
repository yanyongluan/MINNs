from keras.layers import Layer
from keras import backend as K
from keras import activations, initializers, regularizers
from . import pooling_method as pooling

class RC_block(Layer):
    """
    Residual Connection block

    This layer contains a MIL pooling with the layer input to produce a tensor of
    outputs (bag representation residuals).
    This layer is used in MI-Net with RC.

    # Arguments
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, pooling_mode='max', **kwargs):
        self.pooling_mode = pooling_mode
        super(RC_block, self).__init__(**kwargs)

    def call(self, x, mask=None):
        n, d = x.shape

        # do-pooling operator
        output =pooling.choice_pooling(x, self.pooling_mode)

        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)

    def get_config(self):
        config = {'pooling_mode':self.pooling_mode}
        base_config = super(RC_block, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Feature_pooling(Layer):
    """
    Feature pooling layer

    This layer contains a MIL pooling and a FC layer which only has one neural with
    sigmoid activation. The input of this layer is instance features. Via MIL pooling,
    we aggregate instance features to bag features. Finally, we obtain bag score by
    this FC layer with only one neural and sigmoid activation
    This layer is used in MI-Net and MI-Net with DS.

    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, pooling_mode='max', **kwargs):
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Feature_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer,
                                        trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        trainable=True)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape

        # do-pooling operator
        x =pooling.choice_pooling(x, self.pooling_mode)

        # compute bag-level score
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)

        # sigmoid
        output = K.sigmoid(output)

        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias,
            'pooling_mode': self.pooling_mode
        }
        base_config = super(Feature_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Score_pooling(Layer):
    """
    Score pooling layer

    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.

    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, pooling_mode='max', **kwargs):
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Score_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape

        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        x = K.sigmoid(x)

        # do-pooling operator
        output = pooling.choice_pooling(x, self.pooling_mode)

        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias,
            'pooling_mode': self.pooling_mode
        }
        base_config = super(Score_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
