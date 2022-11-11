################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-01
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np
import sys


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """


    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.params = {}
        self.grads = {}

        self.params["weight"] = np.zeros((out_features, in_features))
        self.params["weight"] = self.kaiming_initialisation(self.params["weight"], first_layer=input_layer)
        self.grads["weight"] = np.zeros((out_features, in_features))

        self.params["bias"] = np.zeros((1, out_features))
        self.grads["bias"] = np.zeros((1, out_features))
        #######################
        # END OF YOUR CODE    #
        #######################

    
    def kaiming_initialisation(self, tensor, first_layer = False):
      assert tensor.ndim >= 2


      numerator = 1 if first_layer else 2
      var = np.sqrt(numerator/tensor.shape[1])

      tensor = np.random.normal(0, var, tensor.shape) 
      return tensor



    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.act = x
        out = x @ self.params["weight"].T + self.params["bias"]
        #######################
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

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.grads["weight"] = dout.T @ self.act
        self.grads["bias"] = np.sum(dout, axis = 0)
        dx = dout @  self.params["weight"]
        

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.act = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
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
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        alpha = 1.0
        self.act = x
        out = np.where(x < 0, (alpha * (np.exp(x) - 1)), x)


        #######################
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

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        alpha = 1.0
        dx = dout * np.where(self.act < 0, alpha * np.exp(self.act), np.ones(self.act.shape))
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.act = None  
        #######################
        # END OF YOUR CODE    #
        #######################


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

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        b = np.max(x, axis = 1, keepdims = True)
        numerator = np.exp(x - b)
        denominator = np.sum(numerator, axis = 1, keepdims=True)
        out = numerator/denominator
        self.act = out
        #######################
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

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        prod = np.einsum("ij, ij -> ij", dout, self.act)
        ones = np.ones((self.act.shape[1], self.act.shape[1]))
        prod = prod @ ones
        sub = dout - prod
        dx = np.einsum("ij, ij -> ij", self.act, sub)
        
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.act = None
        #######################
        # END OF YOUR CODE    #
        #######################


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

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = -np.sum(np.log(x[np.arange(y.shape[0]), y])) / x.shape[0]
        #######################
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

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = np.zeros(x.shape)
        dx[np.arange(y.shape[0]), y] = (-1. / (x.shape[0] * x[np.arange(y.shape[0]), y]))

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx