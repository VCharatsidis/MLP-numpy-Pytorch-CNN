"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    mean = 0
    std = 0.0001
    size = (in_features, out_features)

    weights = np.random.normal(mean, std, size)
    biases = np.zeros(out_features)
    grads = np.zeros(out_features)

    self.params = {'weight': weights, 'bias': biases}
    self.grads = {'weight': grads, 'bias': grads}
    self.cache = None
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.#
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = np.dot(x, self.params['weight']) + self.params['bias']
    self.cache = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = self.cache
    dx = np.dot(dout, self.params['weight'].T)

    self.grads['weight'] = np.dot(x.T, dout)
    self.grads['bias'] = np.sum(dout, 0)

    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    zero = np.zeros(x.shape)
    out = np.maximum(x, zero)
    self.cache = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = self.cache
    dx = dout * (x > 0)

    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation. #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    numerator = np.exp(x - np.max(x, axis=1, keepdims=True))
    denominator = np.sum(numerator, axis=1, keepdims=True)

    out = numerator / denominator

    self.cache = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    predictions = self.cache

    softmax_derivative = predictions * (1. - predictions)
    mean = softmax_derivative / predictions.shape[0]

    dx = mean * dout
    s = predictions.reshape(-1, 1)
    der = np.diagflat(s) - np.dot(s, s.T)

    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    # s = predictions.reshape(-1, 1)
    # softmax_derivative = np.diagflat(predictions) - np.dot(s, s.T)
    # #mean = softmax_derivative / predictions.shape[0]
    # dx = mean * dout
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    m = y.shape[0]
    numerator = np.exp(x - np.max(x, axis=1, keepdims=True))
    denominator = np.sum(numerator, axis=1, keepdims=True)

    softmax = numerator / denominator

    log_likelihood = -np.log(softmax[range(m), y])
    out = np.sum(log_likelihood) / m

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    m = y.shape[0]
    numerator = np.exp(x - np.max(x, axis=1, keepdims=True))
    denominator = np.sum(numerator, axis=1, keepdims=True)

    softmax = numerator / denominator
    grad = softmax
    grad[range(m), y] -= 1
    grad = grad / m

    ########################
    # END OF YOUR CODE    #
    #######################

    return grad
