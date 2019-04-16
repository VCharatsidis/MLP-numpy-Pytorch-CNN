import torch
import numpy as np


###################################### one hot pytorch #####################################
batch_size = 200
classes = 10

predictions = torch.LongTensor(batch_size, 1).random_(0, classes)

print("predictions")
print(predictions)

y_onehot = torch.FloatTensor(batch_size, classes)

print(y_onehot)

y_onehot.zero_()
y_onehot.scatter_(1, predictions, 1)

print(y_onehot)

print(predictions[5])
print(y_onehot[5])