import numpy as np
from scipy.special import expit
import sys


# Implémentation du MLP
# Couche d'entrées, 1 couche cachée et couche de sorties
#
# Code similaire à Adaline

class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_output : int
        Number of output units, should be equal to the number of unique class labels.
    n_features : int
        Number of features (dimensions) in the target dataset.Should be equal to the number of columns in the X array.
    n_hidden : int (default: 30)
        Number of hidden units.
    l1 : float (default: 0.0)
        Lambda value for L1-regularization. No regularization if l1=0.0 (default)
    l2 : float (default: 0.0)
        Lambda value for L2-regularization. No regularization if l2=0.0 (default)
    epochs : int (default: 500)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    alpha : float (default: 0.0)
        Momentum constant. Factor multiplied with the gradient of the previous epoch t-1 to improve learning speed
        w(t) := w(t) - (grad(t) + alpha*grad(t-1))
    decrease_const : float (default: 0.0)
        Decrease constant. Shrinks the learning rate after each epoch via eta / (1 + epoch*decrease_const)
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatches : int (default: 1)
        Divides training data into k minibatches for efficiency. Normal gradient descent learning if k=1 (default).
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.

    Attributes
    -----------
    cost_ : list
      Sum of squared errors after each epoch.

    """

    def __init__(self, n_output, n_features,hidden_layers= [30,20,15,10], n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0,
                 decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):

        np.random.seed(random_state)
        self.hidden_layers = hidden_layers
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.W = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]   Target values.

        Returns
        -----------
        onehot : array, shape = (n_labels, n_samples)

        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        "hidden_layers = [50,10]"

        n_layers = len(self.hidden_layers)+1
        W = [0 for i in range(n_layers)]  
        w = np.random.uniform(-1.0, 1.0,
                                size=(self.n_features + 1) * self.hidden_layers[0])
        w = w.reshape(self.hidden_layers[0],self.n_features + 1)
        W[1]= w
        for i in range(2 ,n_layers):
            w = np.random.uniform(-1.0, 1.0,
                               size=self.hidden_layers[i-1] * ( self.hidden_layers[i-2]+ 1))
            w = w.reshape(self.hidden_layers[i-1], self.hidden_layers[i-2]+ 1)
            W[i]= w   


        return W

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Uses scipy.special.expit to avoid overflow
        error for very small input values z.

        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, W):
        """Compute feedforward step

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

        Returns
        ----------
        a1 : array, shape = [n_samples, n_features+1]
            Input values with bias unit.
        z2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer.
        a2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer.
        z3 : array, shape = [n_output_units, n_samples]
            Net input of output layer.
        a3 : array, shape = [n_output_units, n_samples]
            Activation of output layer.

        """
        n_layers = len(self.hidden_layers)+1
        A=[0 for i in range(n_layers+1)]
        Z=[0 for i in range(n_layers+1)]
        A[1] = self._add_bias_unit(X, how='column')
        for i in range(len(self.hidden_layers)):
            if i==0: 
                Z[2] = W[1].dot(A[1].T)
            else :
                Z[i+2] = W[i+1].dot(A[i+1])   

            if i == len(self.hidden_layers) -1:
                A[i+2] = self._sigmoid(Z[i+2])

            else :    
                A[i+2] = self._add_bias_unit(self._sigmoid(Z[i+2]), how='row')

        return A , Z


    def _L2_reg(self, lambda_, W):
        """Compute L2-regularization cost"""
        regularization_cost = 0.0
        for i in range(1,len(W)):
                regularization_cost += np.sum(W[i][:, 1:] ** 2)
        return (lambda_ / 2.0) * regularization_cost                              
    

    def _L1_reg(self, lambda_, W):
        """Compute L2-regularization cost"""
        regularization_cost = 0.0
        for i in range(1,len(W)):
                regularization_cost += np.abs(W[i][:, 1:]).sum()
        return (lambda_ / 2.0) * regularization_cost

    def _get_cost(self, y_enc, output, W):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        output : array, shape = [n_output_units, n_samples]
            Activation of the output layer (feedforward)
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        cost : float
            Regularized cost.

        """
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, W)
        L2_term = self._L2_reg(self.l2, W)
        cost = cost + L1_term + L2_term
        return cost

    #
    # Nous verrons plus tard
    #
    def _get_gradient(self, y_enc, A, Z, W):
        """ Compute gradient step using backpropagation.

        Parameters
        ------------
        a1 : array, shape = [n_samples, n_features+1]
            Input values with bias unit.
        a2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer.
        a3 : array, shape = [n_output_units, n_samples]
            Activation of output layer.
        z2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer.
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        grad1 : array, shape = [n_hidden_units, n_features]
            Gradient of the weight matrix w1.
        grad2 : array, shape = [n_output_units, n_hidden_units]
            Gradient of the weight matrix w2.

        """
        # backpropagation
        
        'hidden_layers= [50,25,15,10]'
        n_layers = len(self.hidden_layers)+1
        grad = [0 for i in range(n_layers)]

        output_layer_idx = n_layers
        sigma = A[output_layer_idx] - y_enc  # erreur de classification
        grad[output_layer_idx-1]= sigma.dot(A[output_layer_idx-1].T)

        Z[output_layer_idx-1]= self._add_bias_unit(Z[output_layer_idx-1], how='row')

        for i in range(output_layer_idx-1 , 1, -1):
            sigma = W[i].T.dot(sigma) * self._sigmoid_gradient(Z[i])
            sigma = sigma[1:, :] #renomer ce sigma peut etre
            if i== 2:
                grad[i-1]= sigma.dot(A[i-1])
            else :     
                grad[i-1]= sigma.dot(A[i-1].T)
                Z[i-1]= self._add_bias_unit(Z[i-1], how='row')

            




        # regularize
        for i in range(1, n_layers):
            grad[i][:, 1:] += self.l2 * W[i][:, 1:]
            grad[i][:, 1:] += self.l1 * np.sign(W[i][:, 1:])


        return grad

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.

        """
        n_layers = len(self.hidden_layers) + 1
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        A ,Z = self._feedforward(X, self.W)
        y_pred = np.argmax(Z[n_layers], axis=0)
        return y_pred

    #
    # Fonction d'entraînement
    #
    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        y : array, shape = [n_samples]
            Target class labels.
        print_progress : bool (default: False)
            Prints progress as the number of epochs
            to stderr.

        Returns:
        ----------
        self

        """
        n_layers = len(self.hidden_layers) + 1
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)  # Vecteur one-hot

        delta_W_prev = [0 for i in range(len(self.W))]
        for i in range(1,len(self.W)):
            delta_W_prev[i] =  np.zeros(self.W[i].shape)
        

        for i in range(self.epochs):  # Nombre de passage sur le dataset

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)  # Permet de réduire le nombre d'epochs nécessaire à la convergence en limitant les risques de "pas" trop grand!

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:  # on mélange le dataset à chaque epoch
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]),
                                  self.minibatches)  # Si le mode minibatch est activé, le dataset en entrée est divisé en batch pour le calcul des gradients
            for idx in mini:
                # feedforward
                A, Z = self._feedforward(X_data[idx], self.W)  # Ce que nous avons vu jusqu'à présent
                cost = self._get_cost(y_enc=y_enc[:, idx], output=A[n_layers], W=self.W)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                #
                # Nous verrons plus en détails
                grad = self._get_gradient(y_enc=y_enc[:, idx], A=A ,Z=Z, W=self.W)
                
                delta_W = [0 for i in range(len(self.W))]
                for i in range(1 , len(self.W)):
                    delta_W[i] = self.eta * grad[i]
                    self.W[i] -= (delta_W[i] + (self.alpha * delta_W_prev[i]))
                    delta_W_prev[i] = delta_W[i]


        return self

# Retour sur le powerpoint







