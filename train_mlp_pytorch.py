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
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 6000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 200

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

  optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)

  loss_fn = nn.CrossEntropyLoss()

  accuracies = []
  losses = []

  for iteration in range(MAX_STEPS_DEFAULT):
      model.train()
      ids = np.random.choice(X_train.shape[0], size=BATCH_SIZE_DEFAULT, replace=False)
      X_train_batch = X_train[ids, :]
      y_train_batch = y_train[ids]

      X_train_batch = np.reshape(X_train_batch, (BATCH_SIZE_DEFAULT, -1))

      X_train_batch = Variable(torch.FloatTensor(X_train_batch))

      output = model.forward(X_train_batch)

      y_train_batch = Variable(torch.LongTensor(y_train_batch))
      loss = loss_fn(output, y_train_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if iteration % EVAL_FREQ_DEFAULT == 0:
        model.eval()
        total_acc = 0
        total_loss = 0

        for i in range(BATCH_SIZE_DEFAULT, len(X_test) + BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT):
            ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))

            x = X_test[ids, :]
            targets = y_test[ids]

            x = np.reshape(x, (BATCH_SIZE_DEFAULT, -1))

            x = Variable(torch.FloatTensor(x))

            pred = model.forward(x)
            acc = accuracy(pred, targets)
            #accuracies.append(acc)
            targets = Variable(torch.LongTensor(targets))
            total_acc += acc
            batch_loss = nn.CrossEntropyLoss()
            calc_loss = batch_loss.forward(pred, targets)
            #losses.append(calc_loss.item())
            total_loss += calc_loss.item()

        denom = len(X_test) / BATCH_SIZE_DEFAULT
        total_acc = total_acc / denom
        total_loss = total_loss / denom
        accuracies.append(total_acc)
        losses.append(total_loss)

        print("total accuracy " + str(total_acc) + " total loss " + str(total_loss))

  plt.plot(accuracies)
  plt.ylabel('accuracies')
  plt.show()

  plt.plot(losses)
  plt.ylabel('losses')
  plt.show()
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