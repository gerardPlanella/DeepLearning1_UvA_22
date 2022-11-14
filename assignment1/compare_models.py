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
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch
import train_mlp_numpy

import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import json

import matplotlib.pyplot as plt
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename, kwargs):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested

    joined_data = {}

    hidden_dims_combinations = kwargs.pop("hidden_dims_combinations")
    use_batch_norm = kwargs.pop("use_batch_norm")

    for hidden_dims in hidden_dims_combinations:
        
        print("Training Pytorch MLP with layers: " + str(hidden_dims))
        _, _, _, logging_info_torch = train_mlp_pytorch.train(**kwargs, hidden_dims=hidden_dims, use_batch_norm=use_batch_norm)
        print("Training Numpy MLP with layers: " + str(hidden_dims))
        _, _, _, logging_info_np = train_mlp_numpy.train(**kwargs, hidden_dims=hidden_dims)
        joined_data["numpy"][str(hidden_dims)] = logging_info_np
        joined_data["torch"][str(hidden_dims)] = logging_info_torch
    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    json_object = json.dumps(joined_data, indent=4)
 
    with open(results_filename, "w") as outfile:
        outfile.write(json_object)

    return joined_data
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename, logging_info):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    if logging_info is None:
        with open(FILENAME, 'r') as openfile:
            logging_info = json.load(openfile)

    assert logging_info is not None

    logging_info_np = logging_info["numpy"]
    logging_info_torch = logging_info["pytorch"]

    fig, axs = plt.subplots(nrows=1, ncols=2)

    fig.suptitle("PyTorch and NumPy MLP Evaluation with different Layer Combinations")

    axs = axs.reshape(-1)

    axs[0].set_title("Training Loss Curve")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epoch")

    axs[1].set_title("Validation Accuracy Curve")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Validation Accuracy")

    
    for model_name in logging_info.keys():
        for combination_name in logging_info[model_name].keys():
            losses = logging_info[model_name][combination_name]["losses"]
            accuracies = []
            x = np.arange(len(losses))
            for epoch in logging_info[model_name][combination_name]["validation"].keys():
                accuracies.append(logging_info[model_name][combination_name]["validation"][epoch])
            axs[0].plot(x, losses, label=model_name + "_" + combination_name)
            axs[1].plot(x, accuracies, label=model_name + "_" + combination_name)
    
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims_combinations', default=[[128], [256, 128], [512, 256, 128]], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

                        
    parser.add_argument('--use_batch_norm', action='store_false',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=0, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    FILENAME = 'results.json' 
    data_dict = None
    if not os.path.isfile(FILENAME):
        data_dict = train_models(FILENAME, kwargs)
    #plot_results(FILENAME, data_dict)