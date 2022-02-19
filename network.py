import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
"""
Wide resnet implementaion
From the paper: "Wide Residual Networks" - https://arxiv.org/pdf/1605.07146.pdf

"""
def conv3x3_net(input_planes,out_planes,stride=1)->nn.Module:
    return nn.Conv2d(input_planes,out_planes,kernel_size=3,padding=1,bias=True,stride=stride)

def conv_initlization(in_net): # in_net is the incoming conv layer
    classname = in_net.__class__.__name__ # get the class name of the layer
    if classname.find('Conv') != -1: # if the layer is a conv layer
        init.xavier_uniform(in_net.weight, gain=np.sqrt(2)) # initialize the weight with xavier uniform
        init.constant(in_net.bias, 0) # initialize the bias with 0
    elif classname.find('BatchNorm') != -1: # if the layer is a batch normalization layer
        init.constant(in_net.weight, 1) # initialize the weight with 1
        init.constant(in_net.bias, 0) # initialize the bias with 0


class wide_block(nn.Module):
    def __init__(self,input_planes,planes,stride=1):
        """
        A wide block is a block with two convolution layers and a shortcut connection which is used in the Wise resnet as refered to in the paper.

        Args:
            input_planes: The number of input planes
            planes: The number of output planes
            stride: The stride of the convolution layer, defaults to 1
        """
        super().__init__()


        # Define layers
        self.batchnorm1 = nn.BatchNorm2d(input_planes)
        self.convolution1 = nn.Conv2d(input_planes,planes,kernel_size=3,padding=1,bias=True)

        self.batchnorm2 = nn.BatchNorm2d(planes)
        self.convolution2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=True)

        self.skipping = nn.Sequential() #shortcut connection

        # Skip connection has to scale dimensions
        if stride !=1 or input_planes != planes: #if stride is not 1 or the number of input planes is not equal to the number of output planes
            self.shortcut= nn.Sequential(nn.Conv2d(input_planes,planes,kernel_size=1,stride=stride,bias=True))

    def forward(self,x)->torch.Tensor:
        x = self.convolution1(F.relu(self.batchnorm1(x))) #First layer with activation and batchnorm
        x = self.convolution2(F.relu(self.batchnorm2(x))) #Second layer with activation and batchnorm
        x += self.skipping(x) #shortcut connection for the resnet
        return x

class Wide_ResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=10):
        super().__init__() # call the parent class constructor
        self.input_planes= 16 # number of input channels
        assert ((depth-4)%6 ==0), 'Depth of the wide-resnet should be 6n+4' # check if the depth is 6n+4
        n_blocks = (depth-4)//6 # number of blocks

        print(f'| Wide-ResNet {depth}x{widen_factor}') # print the network configuration
        nStages = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor] # number of convolutional layers at each stage

        self.convolution1 = conv3x3_net(input_planes=3,out_planes=nStages[0]) # first convolutional layer
        self.L1 = self._wide_layer(wide_block,planes=nStages[1],num_blocks=n_blocks,stride=1) # first wide layer
        self.L2 = self._wide_layer(wide_block,planes=nStages[2],num_blocks=n_blocks,stride=2) # second wide layer
        self.L3 = self._wide_layer(wide_block,planes=nStages[3],num_blocks=n_blocks,stride=2) # third wide layer
        self.batchnorm1 = nn.BatchNorm2d(nStages[3],momentum=0.9) # batchnorm layer
        self.linear = nn.Linear(nStages[3],num_classes) # linear layer

    def _wide_layer(self,block,planes,num_blocks,stride)->nn.Module:
        """
        Args:
            block: the block to be used
            planes: number of output planes
            num_blocks: number of blocks
            stride: stride of the convolutional layer

        Returns:
            a sequential module containing the block layers
        """

        strides = [stride] + [1]*(num_blocks-1) # strides for all blocks
        L = [] # layer list
        for stride in strides: # for each block
            L.append(block(self.input_planes,planes,stride)) # add the block
            self.input_planes = planes # update the number of input channels

        return nn.Sequential(*L) # return the block

    def forward(self,x,get_feat=False)->torch.Tensor:
        """
        x is the input image and get_feat is a boolean variable to get the feature maps
        passing the input image through the network

        """
        x = self.convolution1(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = F.relu(self.batchnorm1(x))
        x = F.avg_pool2d(x,8) # average pooling
        x = x.view(x.size(0),-1) # flatten the output

        if get_feat: # if get_feat is true
            return x # return the output
        else:
            return self.linear(x) # return the output of the linear layer
