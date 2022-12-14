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

"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np


class InversePadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(InversePadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        
        

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right
        
        self.pad = nn.Parameter(torch.randn(1, 3, pad_size, pad_size), requires_grad = True)


        self.device = args.device

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        pad_size = self.pad.shape[2]
        image_width = x.shape[2]
        image_height = x.shape[3]
        prompt = torch.zeros(x.shape)

        origin_h = (image_height -pad_size) // 2
        origin_w = (image_width - pad_size) // 2

        prompt[:, :, origin_h: origin_h + pad_size, origin_w: origin_w + pad_size] = self.pad

        x = x.to(self.device)
        prompt = prompt.to(self.device)
    
        return x + prompt
        #######################
        # END OF YOUR CODE    #
        #######################


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        
        

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right
        
        self.pad_up = nn.Parameter(torch.randn(1, 3, pad_size, image_size), requires_grad = True)
        self.pad_down = nn.Parameter(torch.randn(1, 3, pad_size, image_size), requires_grad = True)

        #Shape of pad_left = (1, 3, h, w), w = pad_size, h = image_size - 2*pad_size
        w = pad_size
        h = image_size - (2*pad_size)
        
        self.pad_left = nn.Parameter(torch.randn(1, 3, h, w), requires_grad = True)
        self.pad_right = nn.Parameter(torch.randn(1, 3, h, w), requires_grad = True)

        self.device = args.device

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        pad_size = self.pad_up.shape[2]
        prompt = torch.zeros(x.shape)
        prompt[:, :, 0:pad_size, :] = self.pad_up
        prompt[:, :, x.shape[2] - pad_size:x.shape[2], :] = self.pad_down

        prompt[:, :, pad_size: x.shape[2] - pad_size, :pad_size] = self.pad_left
        prompt[:, :, pad_size: x.shape[2] - pad_size, x.shape[3] - pad_size:] = self.pad_right

        x = x.to(self.device)
        prompt = prompt.to(self.device)
    
        return x + prompt
        #######################
        # END OF YOUR CODE    #
        #######################


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn
    
        self.patch = nn.Parameter(torch.randn(1, 3, args.prompt_size, args.prompt_size), requires_grad = True)
        self.device = args.device
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.
        patch_size = self.patch.shape[2]
        prompt = torch.zeros(x.shape)
        prompt[:, :, 0:patch_size, 0:patch_size] = self.patch

        x = x.to(self.device)
        prompt = prompt.to(self.device)

        return x + prompt
        #######################
        # END OF YOUR CODE    #
        #######################


class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can be defined as self.patch) of size [prompt_size, prompt_size]
        # that is located at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        self.patch = nn.Parameter(torch.randn(1, 3, args.prompt_size, args.prompt_size), requires_grad = True)
        self.device = args.device
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - Note that, here, you need to place the patch at a random location
        #   and not at the top-left corner.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        patch_size = self.patch.shape[2]
        origin_x = np.random.randint(low=0, high=x.shape[3] - patch_size)
        origin_y = np.random.randint(low=0, high=x.shape[2] - patch_size)
        
        patch_size = self.patch.shape[2]
        prompt = torch.zeros(x.shape)
        prompt[:, :, origin_x:origin_x + patch_size, origin_y: origin_y + patch_size] = self.patch

        x = x.to(self.device)
        prompt = prompt.to(self.device)

        return x + prompt

        #######################
        # END OF YOUR CODE    #
        #######################

