from keras.layers import Layer
from keras import backend as K
from keras import activations, initializations, regularizers
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
        2D tensor with shape: (batch_size, units)
    """
    def __init__(self, pooling_mode='max', **kwargs):
        self.pooling_mode = pooling_mode
        super(RC_block, self).__init__(**kwargs)

    def call(self, x, mask=None):
        n, d = x.shape

        # do-pooling operator
        x =pooling.choice_pooling(x, self.pooling_mode)

        # tile output
        output = K.tile(x, (n,1))

        return output
    
    def get_output_shape_for(self, input_shape):
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
        init: Initializer of the `kernel` weights matrix
        W_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling) 
    
    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (batch_size, units)
    """
    def __init__(self, output_dim, init='glorot_uniform', W_regularizer=None, bias=True, pooling_mode='max', **kwargs):    
        self.init = initializations.get(init)
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode

        self.W_regularizer = regularizers.get(W_regularizer)
        self.bias = bias
        super(Feature_pooling, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))

        if self.bias:
            self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

    def call(self, x, mask=None):
        n, d = x.shape

        # do-pooling operator
        x =pooling.choice_pooling(x, self.pooling_mode)

        # compute bag-level score
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b

        # sigmoid
        output = K.sigmoid(output)

        # tile output
        output = K.tile(output, (n,1))

        return output
    
    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)
    
    def get_config(self):
        config = {'output_dim':self.output_dim, 'init':self.init, 'W_regularizer':self.W_regularizer, 'bias':self.bias, 'pooling_mode':self.pooling_mode}
        base_config = super(Feature_pooling, self).get_config()

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
        init: Initializer of the `kernel` weights matrix
        W_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling) 
    
    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (batch_size, units)
    """
    def __init__(self, output_dim, init='glorot_uniform', W_regularizer=None, bias=True, pooling_mode='max', **kwargs):
        self.init = initializations.get(init)
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode

        self.W_regularizer = regularizers.get(W_regularizer)
        self.bias = bias
        super(Score_pooling, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))

        if self.bias:
            self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

    def call(self, x, mask=None):
        n, d = x.shape

        # compute instance-level score
        x = K.dot(x, self.W)
        if self.bias:
            x += self.b

        # sigmoid
        x = K.sigmoid(x)

        # do-pooling operator
        output = pooling.choice_pooling(x, self.pooling_mode) 

        # tile output
        output = K.tile(output, (n,1))
        return output
    
    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)
    
    def get_config(self):
        config = {'output_dim':self.output_dim, 'init':self.init, 'W_regularizer':self.W_regularizer, 'bias':self.bias, 'pooling_mode':self.pooling_mode}
        base_config = super(Score_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
