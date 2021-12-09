import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


"""Wide resnet implementaion"""

def Conv3x3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=True)

def conv_init(m): # m is a conv layer
    classname = m.__class__.__name__ # get the class name of the layer
    if classname.find('Conv') != -1: # if the layer is a conv layer
        init.xavier_uniform(m.weight, gain=np.sqrt(2)) # initialize the weight with xavier uniform
        init.constant(m.bias, 0) # initialize the bias with 0
    elif classname.find('BatchNorm') != -1: # if the layer is a batch normalization layer
        init.constant(m.weight, 1) # initialize the weight with 1
        init.constant(m.bias, 0) # initialize the bias with 0


class wide_block(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super().__init__()

        self.batchnorm1 = nn.BatchNorm2d(in_planes)
        self.Conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,padding=1,bias=True)

        self.batchnorm2 = nn.BatchNorm2d(planes)
        self.Conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=True)

        self.shortcut = nn.Sequential()

        # Skip connection has to scale dimensions
        if stride !=1 or in_planes != planes: #if stride is not 1 or the number of input planes is not equal to the number of output planes
            self.shortcut= nn.Sequential(nn.Conv2d(in_planes,planes,kernel_size=1,stride=stride,bias=True))

    def forward(self,x):
        out = self.Conv1(F.relu(self.batchnorm1(x))) #First layer with activation and batchnorm
        out = self.Conv2(F.relu(self.batchnorm2(out))) #Second layer with activation and batchnorm
        skip = self.shortcut(x)
        out += self.shortcut(x) #shortcut connection for the resnet
        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=10):
        super().__init__() # call the parent class constructor
        self.in_planes= 16 # number of input channels
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n_blocks = (depth-4)//6 # number of blocks
        k = widen_factor # width factor
        print('| Wide-ResNet %dx%d' %(depth,k)) # print the network configuration
        nStages = [16, 16*k, 32*k, 64*k] # number of convolutional layers at each stage

        self.Conv1 = Conv3x3(in_planes=3,out_planes=nStages[0]) # first convolutional layer
        self.L1 = self._wide_layer(wide_block,planes=nStages[1],num_blocks=n_blocks,stride=1) # first wide layer
        self.L2 = self._wide_layer(wide_block,planes=nStages[2],num_blocks=n_blocks,stride=2) # second wide layer
        self.L3 = self._wide_layer(wide_block,planes=nStages[3],num_blocks=n_blocks,stride=2) # third wide layer
        self.batchnorm1 = nn.BatchNorm2d(nStages[3],momentum=0.9) # batchnorm layer
        self.linear = nn.Linear(nStages[3],num_classes) # linear layer

    def _wide_layer(self,block,planes,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1) # strides for all blocks
        layers = []
        for stride in strides: # for each block
            layers.append(block(self.in_planes,planes,stride)) # add the block
            self.in_planes = planes # update the number of input channels

        return nn.Sequential(*layers) # return the block

    def forward(self,x,get_feat=False):
        out = self.Conv1(x)
        out = self.L1(out)
        out = self.L2(out)
        out = self.L3(out)
        out = F.relu(self.batchnorm1(out))
        out = F.avg_pool2d(out,8) # average pooling
        out = out.view(out.size(0),-1) # flatten the output

        if get_feat: # if get_feat is true
            return out # return the output
        else:
            return self.linear(out) # return the output of the linear layer
