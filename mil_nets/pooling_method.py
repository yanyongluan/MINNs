from keras import backend as K

def max_pooling(x):
    """Max Pooling to obtain aggregation.
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do max-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of max-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = K.max(x, axis=0, keepdims=True)
    return output

def mean_pooling(x):
    """Mean Pooling to obtain aggregation.
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do mean-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of mean-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = K.mean(x, axis=0, keepdims=True)
    return output

def LSE_pooling(x):
    """LSE Pooling to obtain aggregation.
    Do LSE(log-sum-exp) pooling, like LSE(x1, x2, x3) = log(exp(x1)+exp(x2)+exp(x3)).
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do LSE-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of LSE-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = K.log(K.mean(K.exp(x), axis=0, keepdims=True))
    return output

def choice_pooling(x, pooling_mode):
    """Choice the pooling mode
    Parameters
    -------------------
    x : Tensor (N x d)
        Input data to do MIL-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    pooling_mode : string
        Choice the pooling mode for MIL pooling.
    Return
    --------------------
    output : Tensor (1 x d)
            Output of MIL-pooling,
            where d is dimension of instance feature
            (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    if pooling_mode == 'max':
        return max_pooling(x)
    if pooling_mode == 'lse':
        return LSE_pooling(x)
    if pooling_mode == 'ave':
        return mean_pooling(x)
