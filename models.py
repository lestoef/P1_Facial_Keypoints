## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(32*110*110, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    

class Network(nn.Module):
    ''' Template class for the classifier network, inspired by
        https://github.com/udacity/DSND_Term1/blob/1196aafd48a2278b02eff85510b582fd7e2a9d2d/lessons/DeepLearning/new-intro-to-pytorch/fc_model.py
    '''
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.25):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        return x  # F.log_softmax(x, dim=1)


class NaimishNet(nn.Module):
    ''' inspired by https://arxiv.org/pdf/1710.00977.pdf
        1. This network takes in a square (same width and height), grayscale image as input
        2. It ends with a linear layer that represents the 68 keypoints
    '''
    def __init__(self):
        super(NaimishNet, self).__init__()
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # output size = (W-F)/S + 1 
        self.conv1 = nn.Conv2d(1, 32, 4) # output_size after pooling = ((224-4)/1 + 1) // 2 = 110
        self.conv2 = nn.Conv2d(32, 64, 3) # output size = ((110-3/1 + 1) // 2 = 54
        self.conv3=nn.Conv2d(64, 128, 2) # output size = ((54-2)/1 + 1) // 2 = 26
        self.conv4=nn.Conv2d(128, 256, 1) # output size = ((26-1)/1 + 1) // 2 = 13
        
        # 256 outputs from the last convolutional layer * 13*13 from the pooling layer
        self.fc1 = nn.Linear(256*13*13, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 2*68)
       
        # dropout layer with fixed probability
        self.dropout=nn.Dropout(0.3)
        
    def forward(self, x):
        
        # only the last activation function is linear, all other are exponential 
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # three linear layers with dropout in between
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
class NaimishNet2(nn.Module):

    def __init__(self):
        super(NaimishNet2, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(3, 64, 5) # ((224-5)/1 + 1) // 2 = 110
        self.conv2 = nn.Conv2d(64, 128, 3) # ((110-3)/1 + 1) // 2 = 59
        self.conv3 = nn.Conv2d(128, 256, 3) # ((59-3)/1 + 1) // 2 = 28
        self.conv4 = nn.Conv2d(256, 512, 3) # ((28-3)/1 + 1) // 2 = 13
        self.conv5 = nn.Conv2d(512, 1024, 1) # ((13-1)/1 + 1) // 2 = 6

        self.fc1 = nn.Linear(1024*6*6, 4608)
        self.fc2 = nn.Linear(4608, 4608)
        self.fc3 = nn.Linear(4608, 68*2)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
