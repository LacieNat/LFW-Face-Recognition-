import os
#import matplotlib.pyplot as plt
#%pylab inline
import numpy as np
import pickle

import csv

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam, sgd, nesterov_momentum
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne import PrintLayerInfo


X_train = np.load('X_train.npy').reshape(-1, 1, 50, 37).astype(np.float32)
y_train = np.load('y_train.npy').astype(np.int32)
X_test = np.load('X_test.npy').reshape(-1, 1, 50, 37).astype(np.float32)



layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, 1, 50, 37)}),

    # first stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # second stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 128}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 128}),

    # the output layer
    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]

net0 = NeuralNet(
    layers=layers0,
    max_epochs=50,

    update=adam,
    update_learning_rate=0.0002,
#     update_momentum=0.9,
    objective_l2=0.0025,

    train_split=TrainSplit(eval_size=0.1),
    verbose=2,
)

# net0.initialize()
# layer_info = PrintLayerInfo()
# layer_info(net0)
net0.fit(X_train, y_train)
result = net0.predict(X_test)
np.save('cnn_res.npy', result)


#fName = "./features/cnn.pkl"
#print("Saving CNN to ./features/cnn.pkl")
#with open(fName, "w") as f:
#    pickle.dump(net0, f)
