"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
from mlp_pytorch import MLP
import torch.nn as nn
import cifar10_utils
from torch.autograd import Variable

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 10000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # classes = 10
  #
  # predictions = predictions.detach().numpy()
  #
  # predictions = np.argmax(predictions, 1)
  # print(predictions)
  # print(targets)
  # predictions_one_hot = list()
  # for value in predictions:
  #     letter = [0 for _ in range(10)]
  #     letter[value] = 1
  #     predictions_one_hot.append(letter)
  #
  # predictions_one_hot = np.array(predictions_one_hot)
  #
  # print(targets.shape)
  # input()
  # result = predictions_one_hot == targets
  # sum = np.sum(result)
  # accuracy = sum / float(targets.shape[0])

  predictions = predictions.detach().numpy()
  preds = np.argmax(predictions, 1)
  result = preds == targets
  sum = np.sum(result)
  accuracy = sum / float(targets.shape[0])

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  classes = 10
  input_dim = 3 * 32 * 32

  X_train_raw, y_train_raw, X_test_raw, y_test_raw = cifar10_utils.load_cifar10(cifar10_utils.CIFAR10_FOLDER)
  X_train, y_train, X_test, y_test = cifar10_utils.preprocess_cifar10_data(X_train_raw, y_train_raw, X_test_raw,
                                                                           y_test_raw)

  model = MLP(input_dim, dnn_hidden_units, classes)
  print(model)

  model_params = list(model.parameters())
  optimizer = torch.optim.Adam(model_params, lr=LEARNING_RATE_DEFAULT)
  loss_fn = nn.CrossEntropyLoss()

  model.train()
  train_losses = []
  valid_losses = []
  for iteration in range(MAX_STEPS_DEFAULT):
      ids = np.random.choice(X_train.shape[0], size=BATCH_SIZE_DEFAULT, replace=False)
      X_train_batch = X_train[ids, :]
      y_train_batch = y_train[ids]

      X_train_batch = np.reshape(X_train_batch, (BATCH_SIZE_DEFAULT, -1))

      X_train_batch = Variable(torch.FloatTensor(X_train_batch))

      output = model.forward(X_train_batch)

      y_train_batch = Variable(torch.LongTensor(y_train_batch))
      loss = loss_fn(output, y_train_batch)
      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

      if iteration % EVAL_FREQ_DEFAULT == 0:
          ids = np.random.choice(X_test.shape[0], size=BATCH_SIZE_DEFAULT, replace=False)
          X_test_batch = X_test[ids, :]
          y_test_batch = y_test[ids]

          X_test_batch = np.reshape(X_test_batch, (BATCH_SIZE_DEFAULT, -1))

          X_test_batch = Variable(torch.FloatTensor(X_test_batch))

          output = model.forward(X_test_batch)

          acc = accuracy(output, y_test_batch)
          print(acc)


  print(train_losses[-1])
  model.eval()


  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()