# Based on code from: http://cs231n.github.io/assignments2018/assignment1/
import numpy as np
import os
from six.moves import cPickle as pickle
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
  """
  Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
  it for the two-layer neural net classifier. These are the same steps as
  we used for the SVM, but condensed to a single function.
  """
  # Load the raw CIFAR-10 data
  cifar10_dir = 'cifar10/cifar-10-batches-py'

  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # Subsample the data
  mask = list(range(num_training, num_training + num_validation))
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = list(range(num_training))
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = list(range(num_test))
  X_test = X_test[mask]
  y_test = y_test[mask]

  # subtract mean image
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image

  # normalize pixel distributions over channels
  mean_channels = np.ones((32, 32, 3)) * np.mean(X_train, axis=(0, 1, 2))
  std_channels = np.ones((32, 32, 3)) * np.std(X_train, axis=(0, 1, 2))
  X_train -= mean_channels
  X_val -= mean_channels
  X_test -= mean_channels

  X_train /= std_channels
  X_val /= std_channels
  X_test /= std_channels


  # Reshape data to rows
  X_train = X_train.reshape(num_training, -1)
  X_val = X_val.reshape(num_validation, -1)
  X_test = X_test.reshape(num_test, -1)

  # Package data into a dictionary
  return {
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val,
    'X_test': X_test, 'y_test': y_test,
  }
    


