"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
    self.batchNorm1 = nn.BatchNorm2d(64)
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.batchNorm2 = nn.BatchNorm2d(128)
    self.relu2 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv3_a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.batchNorm3_a = nn.BatchNorm2d(256)
    self.relu3_a = nn.ReLU()

    self.conv3_b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.batchNorm3_b = nn.BatchNorm2d(256)
    self.relu3_b = nn.ReLU()

    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv4_a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.batchNorm4_a = nn.BatchNorm2d(512)
    self.relu4_a = nn.ReLU()

    self.conv4_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.batchNorm4_b = nn.BatchNorm2d(512)
    self.relu4_b = nn.ReLU()

    self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv5_a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.batchNorm5_a = nn.BatchNorm2d(512)
    self.relu5_a = nn.ReLU()

    self.conv5_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.batchNorm5_b = nn.BatchNorm2d(512)
    self.relu5_b = nn.ReLU()

    self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
    self.linear = nn.Linear(512, n_classes)
    self.softmax = nn.Softmax(dim=1)
    #
    #
    #
    # self.layers = nn.Sequential(
    #   self.conv1,
    #   self.batchNorm1,
    #   self.relu1,
    #   self.maxpool1,
    #
    #   self.conv2,
    #   self.batchNorm2,
    #   self.relu2,
    #   self.maxpool2,
    #
    #   self.conv3_a,
    #   self.batchNorm3_a,
    #   self.relu3_a,
    #
    #   self.conv3_b,
    #   self.batchNorm3_b,
    #   self.relu3_b,
    #
    #   self.conv4_a,
    #   self.batchNorm4_a,
    #   self.relu4_a,
    #
    #   self.conv4_b,
    #   self.batchNorm4_b,
    #   self.relu4_b,
    #
    #   self.maxpool4,
    #
    #   self.conv5_a,
    #   self.batchNorm5_a,
    #   self.relu5_a,
    #
    #   self.conv5_b,
    #   self.batchNorm5_b,
    #   self.relu5_b,
    #
    #   self.maxpool5,
    #   self.avgpool,
    #   self.linear,
    #   self.softmax
    # )

    self.layers = nn.Sequential(
      nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

      nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

      nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),

      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),

      nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

      nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
      nn.AvgPool2d(kernel_size=(1, 1), stride=1, padding=0),

      nn.Linear(512, n_classes)

    )

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

   # print(x.shape)
   #  out = self.conv1(x)
   # # print(out.shape)
   #  out = self.batchNorm1(out)
   #  #print(out.shape)
   #  out = self.relu1(out)
   #  #print(out.shape)
   #  out = self.maxpool1(out)
   #  #print(out.shape)
   #
   # # print("conv2")
   #  out = self.conv2(out)
   #  #print(out.shape)
   #  out = self.batchNorm2(out)
   # # print(out.shape)
   #  out = self.relu2(out)
   # # print(out.shape)
   #  out = self.maxpool2(out)
   #  #print(out.shape)
   #
   # # print("conv 3_a")
   #  out = self.conv3_a(out)
   # # print(out.shape)
   #  out = self.batchNorm3_a(out)
   #  #print(out.shape)
   #  out = self.relu3_a(out)
   #  #print(out.shape)
   #
   #  #print("conv 3_b")
   #  out = self.conv3_b(out)
   #  #print(out.shape)
   #  out = self.batchNorm3_b(out)
   #  #print(out.shape)
   #  out = self.relu3_b(out)
   #  #print(out.shape)
   #
   #  out = self.maxpool3(out)
   # # print("max pool 3")
   #  #print(out.shape)
   #
   # # print("conv 4_a")
   #  out = self.conv4_a(out)
   # # print(out.shape)
   #  out = self.batchNorm4_a(out)
   #  #print(out.shape)
   #  out = self.relu4_a(out)
   #  #print(out.shape)
   #
   # # print("conv 4_b")
   #  out = self.conv4_b(out)
   #  #print(out.shape)
   #  out = self.batchNorm4_b(out)
   # # print(out.shape)
   #  out = self.relu4_b(out)
   # # print(out.shape)
   #
   #  out = self.maxpool4(out)
   #
   # # print("conv 5_a")
   #  out = self.conv5_a(out)
   #  #print(out.shape)
   #  out = self.batchNorm5_a(out)
   #  #print(out.shape)
   #  out = self.relu5_a(out)
   # # print(out.shape)
   #
   #  #print("conv 5_b")
   #  out = self.conv5_b(out)
   #  #print(out.shape)
   #  out = self.batchNorm5_b(out)
   #  #print(out.shape)
   #  out = self.relu5_b(out)
   #  #print(out.shape)
   #
   #  out = self.maxpool5(out)
   #
   #  out = self.avgpool(out)
   #  # print("avg pool")
   #  # print(out.shape)
   #
   #  out = out.view(out.shape[0], -1)
   #  # print("flaten")
   #  # print(out.shape)
   #  out = self.linear(out)
    # print("linear ")
    # print(out.shape)

    out = x
    for layer in self.layers:
        if isinstance(layer, nn.Linear):
            out = out.view(out.shape[0], -1)

        out = layer.forward(out)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
