import numpy as np
import scipy

class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        print("start of knn algorithm")
        self.k = k
        self.X_train = []
        self.y_train = []
    
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y
    
    def find_dist(self, X_test):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train.

        Hint : Use scipy.spatial.distance.cdist

        Returns :
        - dist_ : Distances between each test point and training point
        """
        length = len(self.X_train)
        # store a list of list of distance
        dist_ = []
        for single_test in X_test:
            dist = scipy.spatial.distance.cdist(self.X_train, [single_test], 'euclidean')
            dist_.append(dist)

        return dist_
    
    def predict(self, X_test):
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        dist_ = self.find_dist(X_test)
        pred = []
        for one_list in dist_:
            one_list = np.array(one_list.T[0], dtype = object)
            idx = np.argpartition(one_list, self.k)
            nearest_labels = list(self.y_train[idx[:self.k]])
            predicted = max(set(nearest_labels), key=nearest_labels.count)
            pred.append(predicted)
        return pred