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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, batch_size = 128, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

 
    out = np.zeros((batch_size, num_classes))
    confusion_matrix = np.zeros((num_classes, num_classes))
    n_batches = 0

    for batch_num, (x, t) in enumerate(data_loader):
      x_flat = x.reshape(x.shape[0], -1)
      t_flat = t.reshape(-1, 1)

      out[batch_num, :] = model.forward(x_flat)
      confusion = confusion_matrix(out[batch_num, :], t_flat)
      confusion_matrix += confusion
      n_batches+=1
      
    confusion_matrix = confusion_matrix // n_batches
    metrics = confusion_matrix_to_metrics(confusion_matrix)
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train_model_SGD(model, loss_module, data_loader, lr):
  # x: (batch_size, 3, 32, 32)
  # t: (batch_size, )
  losses = []
  for batch_num, (x, t) in enumerate(data_loader):
    x_flat = x.reshape(x.shape[0], -1)
    t_flat = t.reshape(-1, 1)

    y = model.forward(x_flat)
    loss = loss_module.forward(y)
    losses.append(loss)

    dx = loss_module.backward(y, t_flat)

    model.backward(dx)
    for module in model.modules:
      if isinstance(module, LinearModule):
        module.params["weight"] -= lr*module.params["weight"]
        module.params["bias"] -= lr*module.params["bias"]

  return model, losses





def train(hidden_dims, lr, batch_size, epochs, seed, data_dir, input_size = 32*32*3, n_classes = 10):
  """
  Performs a full training cycle of MLP model.

  Args:
    hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
    lr: Learning rate of the SGD to apply.
    batch_size: Minibatch size for the data loaders.
    epochs: Number of training epochs to perform.
    seed: Seed to use for reproducible results.
    data_dir: Directory where to store/find the CIFAR10 dataset.
  Returns:
    model: An instance of 'MLP', the trained model that performed best on the validation set.
    val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                    validation set per epoch (element 0 - performance after epoch 1)
    test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                    performed best on the validation. Between 0.0 and 1.0
    logging_info: An arbitrary object containing logging information. This is for you to 
                  decide what to put in here.

  TODO:
  - Implement the training of the MLP model. 
  - Evaluate your model on the whole validation set each epoch.
  - After finishing training, evaluate your model that performed best on the validation set, 
    on the whole test dataset.
  - Integrate _all_ input arguments of this function in your training. You are allowed to add
    additional input argument if you assign it a default value that represents the plain training
    (e.g. '..., new_param=False')

  Hint: you can save your best model by deepcopy-ing it.
  """

  # Set the random seeds for reproducibility
  np.random.seed(seed)
  torch.manual_seed(seed)

  ## Loading the dataset
  cifar10 = cifar10_utils.get_cifar10(data_dir)
  cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                return_numpy=True)

  #######################
  # PUT YOUR CODE HERE  #
  #######################

  # TODO: Initialize model and loss module
  model = MLP(input_size, hidden_dims, n_classes)
  loss_module = CrossEntropyModule()
  # TODO: Training loop including validation
  val_metrics = []
  val_accuracies = []
  model.clear_cache()
  for epoch in range(epochs):
    model, losses = train_model_SGD(model, loss_module, cifar10_loader["train"], lr)
    val_metrics.append(evaluate_model(model, cifar10_loader["validation"], n_classes))
    val_accuracies.append(val_metrics["accuracy"])
  # TODO: Test best model
  test_metrics  = evaluate_model(model, cifar10_loader["test"], n_classes)
  test_accuracy = test_metrics["accuracy"]
  # TODO: Add any information you might want to save for plotting
  info = {
    "input_size": input_size,
    "batch_size": batch_size,
    "lr": lr,
    "epochs": epochs,
    "n_classes": n_classes
    }
  logging_info = {
    "info": info,
    "train": {"losses": losses},
    "validation": val_metrics, 
    "test": test_metrics
    }
  #######################
  # END OF YOUR CODE    #
  #######################

  return model, val_accuracies, test_accuracy, logging_info



if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    