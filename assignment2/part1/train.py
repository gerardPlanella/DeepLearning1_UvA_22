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
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from tqdm import tqdm
from copy import deepcopy
import datetime
import pickle
from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Randomly initialize and modify the model's last layer for CIFAR100.
    
    fc_features_in = model.fc.in_features
    fc_features_out = num_classes

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(fc_features_in, fc_features_out)
    model.fc.weight = nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)    
    model.fc.bias = nn.init.zeros_(model.fc.bias)


    #######################
    # END OF YOUR CODE    #
    #######################

    return model

def train_step_model(device, model, criterion, optimizer, data):
  losses = []

  model.train()
  for x, t in data:
    x = x.to(device)
    t = t.to(device)

    optimizer.zero_grad()
    
    preds = model(x)

    loss = criterion(preds, t)
    losses.append(loss.cpu().detach().numpy())

    loss.backward()
    optimizer.step()

  return model, np.mean(losses)


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_data, validation_data = get_train_validation_set(data_dir, augmentation_name=augmentation_name)

    trainLoader = torch.utils.data.DataLoader(
        train_data, batch_size, shuffle=True, drop_last = True)


    valLoader = torch.utils.data.DataLoader(
        validation_data, batch_size, shuffle=True, drop_last = False)

    data = {
        "train":trainLoader,
        "val":valLoader
    }
    # Initialize the optimizer (Adam) to train the last layer of the model.

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    # Training loop with validation after each epoch. Save the best model.
    best_model_state_dict = None
    best_acc = 0
    best_epoch = 0
    losses = []
    acc = []

    for epoch in tqdm(range(epochs)):
        model, loss = train_step_model(device, model, criterion, optimizer, data["train"])
        losses += [loss]


        accuracy = evaluate_model(model, data["val"], device)
        acc += [accuracy]

        print(f" Loss: {loss}, Accuracy: {accuracy}")

        if(accuracy > best_acc):
            best_acc = accuracy
            best_epoch = epoch
            best_model_state_dict = deepcopy(model.state_dict())
        
    model.load_state_dict(best_model_state_dict)
    checkpoint_name+=f"_epoch={best_epoch}_val_acc={best_acc}.pt"
    torch.save(best_model_state_dict, data_dir+checkpoint_name)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    acc = 0

    with torch.no_grad():
        for x, t in data_loader:
            x = x.to(device)
            t = t.to(device)
            out = model(x)
            
            out = out.cpu().detach().numpy()
            preds = np.argmax(out, axis = 1)

            acc += np.sum(t.cpu().numpy() == preds)

        

    accuracy =  (acc / len(data_loader.dataset)) * 100

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if(torch.cuda.is_available()):
        print("Using GPU: " + torch.cuda.get_device_name(device))
    else:
        print("Using CPU")

    # Load the model
    model = get_model()

    # Train the model
    checkpoint_name = datetime.datetime.now().strftime(f"%Y_%d_%m_%M_%S_lr={lr}_augmentations={augmentation_name}_batchSize={batch_size}")
    train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name)

    # Evaluate the model on the test set
    test_data = get_test_set()
    testLoader = torch.utils.data.DataLoader(
      test_data, batch_size, shuffle=True, drop_last = False)
    
    accuracy = evaluate_model(model, testLoader, device)

    print("Obtained Testing Accuracy of: " + str(accuracy) + "%")
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
