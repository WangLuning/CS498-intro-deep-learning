import numpy as np
import scipy
from operator import add, sub

class Perceptron():
    def __init__(self):
        """
        Initialises Perceptron classifier with initializing 
        weights, alpha(learning rate) and number of epochs.
        """
        self.w = None
        self.alpha = 0.5
        self.epochs = 5
        
    def train(self, X_train, y_train):
        """
        Train the Perceptron classifier. Use the perceptron update rule
        as introduced in Lecture 3.

        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        D = len(X_train[0])
        print("dimension is: " + str(D))
        self.w = np.zeros((10, D))

        for times in range(self.epochs):
            print("epoch " + str(times))
            for i in range(X_train.shape[0]):
                train = X_train[i]
                #print(train.shape)
                predicted_one = np.argmax(np.dot(self.w, train.T))
                if predicted_one != y_train[i]:
                    change = [data * (predicted_one - y_train[i]) * self.alpha for data in X_train[i]]
                    self.w[y_train[i]] = list(map(add, self.w[y_train[i]], change))

    def predict(self, X_test):
        """
        Predict labels for test data using the trained weights.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        print("start predicting")
        pred = []
        for test in X_test:
            predicted = max(np.dot(self.w, test.T))
            pred.append(predicted)
        return pred