import numpy as np


class NeuralNetwork:
    """
    A multi-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices.

    The network uses a nonlinearity after each fully connected layer except for the
    last. You will implement two different non-linearities and try them out: Relu
    and sigmoid.

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_sizes, output_size, num_layers, nonlinearity='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H_1)
        b1: First layer biases; has shape (H_1,)
        .
        .
        Wk: k-th layer weights; has shape (H_{k-1}, C)
        bk: k-th layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: List [H1,..., Hk] with the number of neurons Hi in the hidden layer i.
        - output_size: The number of classes C.
        - num_layers: Number of fully connected layers in the neural network.
        - nonlinearity: Either relu or sigmoid
        """
        self.num_layers = num_layers

        assert(len(hidden_sizes)==(num_layers-1))
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params['W' + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params['b' + str(i)] = np.zeros(sizes[i])

        print('in the init process the params')
        print(self.params.keys())
        print('dimension of input')
        print(input_size)
        print('classes of output')
        print(output_size)

        if nonlinearity == 'sigmoid':
            self.nonlinear = sigmoid
            self.nonlinear_grad = sigmoid_grad
        elif nonlinearity == 'relu':
            self.nonlinear = relu
            self.nonlinear_grad = relu_grad

        # my defined variables
        self.output_size = output_size


    def forward(self, X):
        """
        Compute the scores for each class for all of the data samples.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.

        Returns:
        - scores: Matrix of shape (N, C) where scores[i, c] is the score for class
            c on input X[i] outputted from the last layer of your network.
        - layer_output: Dictionary containing output of each layer BEFORE
            nonlinear activation. You will need these outputs for the backprop
            algorithm. You should set layer_output[i] to be the output of layer i.

        """
        num_of_data = X.shape[0]
        dimension = X.shape[1]

        scores = np.zeros((num_of_data, len(self.params['b' + str(self.num_layers)])))
        assert(scores.shape[1] == self.output_size)

        layer_output = {}
        #############################################################################
        # TODO: Write the forward pass, computing the class scores for the input.   #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C). Store the output of each layer BEFORE nonlinear activation  #
        # in the layer_output dictionary                                            #
        #############################################################################
        layer_input = X

        for i in range(1, len(self.params) // 2 + 1):
            layer_input = np.dot(layer_input, self.params['W' + str(i)]) + self.params['b' + str(i)]
            # store the output of each layer before activation function
            layer_output['o' + str(i)] = layer_input
            layer_input = self.nonlinear(layer_input)
        # it is the final score after all the calculation
        scores = layer_input

        return scores, layer_output

    def select_scores(self, scores):
        selected_scores = []
        for i in range(scores.shape[0]):
            selected_scores.append(np.argmax(scores[i, :]))
        return selected_scores

    def backprop(self, X, y, output):
        if self.num_layers == 2:
            self.two_layer_backprop(X, y, output)
        elif self.num_layers == 3:
            self.three_layer_output(X, y, output)

    def two_layer_backprop(self, X, y, output):
        _, layer_output = self.forward(X)
        # organize the ground truth y into a matrix for further calculation
        gt = np.zeros(output.shape)
        for i in range(len(y)):
            gt[i,y[i]] = 1
        # 1
        last_loss = 1 / X.shape[0] * (output - gt)
        # 2
        layer2_error = np.multiply(last_loss, output)
        # 3
        output1_activate = self.nonlinear(layer_output['o1'])
        dev_w2 = output1_activate.T.dot(layer2_error)
        # 4
        dev_b2 = np.sum(layer2_error, 0)
        # 5 layer2_error W2 output1_activate
        dev_j = np.multiply(layer2_error.dot(self.params['W2'].T), sigmoid_grad(output1_activate))
        # 6
        dev_w1 = X.T.dot(dev_j)
        # 7
        dev_b1 = np.sum(dev_j, 0)

        # update the parameters
        self.params['W1'] -= dev_w1
        self.params['W2'] -= dev_w2
        self.params['b1'] -= dev_b1
        self.params['b2'] -= dev_b2

    def train_one(self,X, y):
        scores, layer_output = self.forward(X)
        #selected_scores = self.select_scores(scores)
        self.backprop(X, y, scores)

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        for i in range(iterations_per_epoch):
            batch_indicies = np.random.choice(num_train, batch_size, replace = False)
            X_batch = X[batch_indicies]
            y_batch = y[batch_indicies]
            scores, output = self.forward(X_batch)
            selected_scores = self.select_scores(scores)
            print("Loss: \n" + str(np.mean(np.square(y_batch - selected_scores))))
            self.train_one(X_batch, y_batch)

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        scores, layer_output = self.forward(X)
        selected_scores = self.select_scores(scores)
        ###########################################################################
        # TODO: Implement classification prediction. You can use the forward      #
        # function you implemented                                                #
        ###########################################################################
        #print('predicted output')
        #print(selected_scores)
        return selected_scores


def sigmoid(X):
    #############################################################################
    # TODO: Write the sigmoid function                                          #
    #############################################################################
    return 1 / (1 + np.exp(-X))

def sigmoid_grad(X):
    #############################################################################
    # TODO: Write the sigmoid gradient function                                 #
    #############################################################################
    return np.multiply(X, (1 - X))

def relu(X):
    #############################################################################
    #  TODO: Write the relu function                                            #
    #############################################################################
    return np.maximum(X, 0)

def relu_grad(X):
    #############################################################################
    # TODO: Write the relu gradient function                                    #
    #############################################################################
    X[X<=0] = 0
    X[X>0] = 1
    return X
