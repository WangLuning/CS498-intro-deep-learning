import numpy as np

class Softmax():
    def __init__(self):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 0.5
        self.epochs = 300
        self.reg_const = 0.05
    
    def calc_gradient(self, X_train, y_train):
        """
        Calculate gradient of the softmax loss
          
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing a minibatch of data.
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.

        Returns:
        - gradient with respect to weights W; an array of same shape as W
        """
        dimension, num_of_classes = self.w.shape
        num_of_data = X_train.shape[0]
        grad_w = np.zeros((dimension, num_of_classes))

        for i in range(num_of_data):
            score_one = X_train[i].dot(self.w)
            score_one -= np.max(score_one)
            # using softmax prob but not the simple direct prob
            exp_sum = np.sum(np.exp(score_one))
            softmax_prob = np.exp(score_one[y_train[i]]) / exp_sum
            softmax_all = np.exp(score_one) / exp_sum
            for pos in range(num_of_classes):
                if pos == y_train[i]:
                    dscore = softmax_all[pos] - 1
                else:
                    dscore = softmax_all[pos]
                grad_w[:, pos] += dscore * X_train[i]

        grad_w /= num_of_data
        grad_w += self.reg_const * self.w
        return grad_w
    
    def train(self, X_train, y_train):
        """
        Train Softmax classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;
        
        Hint : Operate with Minibatches of the data for SGD
        """
        print('start training')
        num_of_data, dimension = X_train.shape
        # we'd better hope the training labels contain everything we need...
        num_of_classes = np.max(y_train) + 1
        size_of_batch = 100

        # give an init weight for the first round
        if self.w is None:
            self.w = 0.01 * np.random.randn(dimension, num_of_classes)

        for i in range(self.epochs):
            indices = np.random.choice(num_of_data, size_of_batch)
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            grad_w = self.calc_gradient(X_batch, y_batch)

            self.w -= self.alpha * grad_w
    
    def predict(self, X_test):
        """
        Use the trained weights of softmax classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        print('start predicting')
        pred = np.argmax(X_test.dot(self.w), axis = 1)
        return pred 