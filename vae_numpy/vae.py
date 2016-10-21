"""
This implementation is based on y0ast code (2014 version)
https://github.com/y0ast/Variational-Autoencoder/tree/5c06a7f14de7f872d837cd4268ee2d081a90056d
"""

import time
import numpy as np
from scipy.misc import logsumexp

class VA:
    """Stochastic Gradient Variational Bayes

    An auto-encoder with variational Bayes inference.

    Parameters
    ----------
    n_components_decoder : int, optional
        Number of binary hidden units for decoder.

    n_components_encoder : int, optional
        Number of binary hidden units for encoder.

    n_hidden_variables : int, optional
        The dimensionality of Z

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    sampling_rounds : int, optional
        Number of sampling rounds done on the minibatch

    continuous : boolean, optional
        Set what type of data the auto-encoder should model

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.

    Attributes
    ----------
    'params' : list-like, list of weights and biases.


    Examples
    --------

    ----------
    References

    [1] Kingma D.P., Welling M. Stochastic Gradient VB and the Variational Auto-Encoder
    Arxiv, preprint. http://arxiv.org/pdf/1312.6114v6.pdf
    """

    def __init__(self, n_components_decoder=200, n_components_encoder=200,
                 n_hidden_variables=20, learning_rate=0.01, batch_size=100,
                 n_iter=10, sampling_rounds=1, continuous=False, verbose=False):
        self.n_components_decoder = n_components_decoder
        self.n_components_encoder = n_components_encoder
        self.n_hidden_variables = n_hidden_variables

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.sampling_rounds = sampling_rounds
        self.verbose = verbose

        self.continuous = continuous
        
        self.initialized = False

    def _initParams(self, dimX):
        """Create all weight and bias parameters with the right dimensions

        Parameters
        ----------
        dimX : scalar
            The dimensionality of the input data X
        """
        sigmaInit = 0.01
        W1 = np.random.randn(self.n_components_encoder, dimX) * sigmaInit
        b1 = np.random.randn(self.n_components_encoder, 1) * sigmaInit

        W2 = np.random.randn(self.n_hidden_variables, self.n_components_encoder) * sigmaInit
        b2 = np.random.randn(self.n_hidden_variables, 1) * sigmaInit

        W3 = np.random.randn(self.n_hidden_variables, self.n_components_encoder) * sigmaInit
        b3 = np.random.randn(self.n_hidden_variables, 1) * sigmaInit

        W4 = np.random.randn(self.n_components_decoder, self.n_hidden_variables) * sigmaInit
        b4 = np.random.randn(self.n_components_decoder, 1) * sigmaInit

        W5 = np.random.randn(dimX, self.n_components_decoder) * sigmaInit
        b5 = np.random.randn(dimX, 1) * sigmaInit

        self.params = {"W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5,
                        "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5}

        if self.continuous:
            W6 = np.random.randn(dimX, self.n_components_decoder) * sigmaInit
            b6 = np.random.randn(dimX, 1) * sigmaInit
            self.params.update({"W6": W6, "b6": b6})

        # now initialise adam storage
        self.gradMean = dict()
        self.gradVar = dict()
        for key in self.params:
            self.gradMean[key] = 0.00
            self.gradVar[key] = 0.00
        self.betaMean = 0.9
        self.betaVar = 0.99
        self.epsi = 10e-8
        self.t = 0
        
        self.initialized = True

    def _lowerbound(self, x, K = 1):
        """
        Compute the lowerbound
        """
        # define network
        
        W1, W2, W3, W4, W5 = self.params["W1"], self.params["W2"], self.params["W3"],\
        self.params["W4"], self.params["W5"]
        b1, b2, b3, b4, b5 = self.params["b1"], self.params["b2"], self.params["b3"],\
        self.params["b4"], self.params["b5"]

        if self.continuous:
            W6, b6 = self.params["W6"], self.params["b6"]
            activation = lambda x: np.log(1 + np.exp(x))
        else:
            activation = lambda x: np.tanh(x,x)

        sigmoid = lambda x: 1. / (1 + np.exp(-x))        
        
        # IWAE: first replicate the input
        N = x.shape[1]
        #x = np.repeat(x, K, axis = 1)
        #x = np.tile(x.reshape(-1, 1), (1, K)).reshape(x.shape[0], K * N)       
        x = np.tile(x, (1, K))

        # compute forward pass
        h_encoder = activation(W1.dot(x) + b1)

        mu_encoder = W2.dot(h_encoder) + b2
        log_sigma_encoder = 0.5 * (W3.dot(h_encoder) + b3)


        eps = np.random.randn(self.n_hidden_variables, x.shape[1])
        z = mu_encoder + np.exp(log_sigma_encoder) * eps

        h_decoder = activation(W4.dot(z) + b4)

        y = sigmoid(W5.dot(h_decoder) + b5)

        if self.continuous:
            log_sigma_decoder = 0.5 * (W6.dot(h_decoder) + b6)
            logpxz = np.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_decoder) -
                            0.5 * ((x - y) / np.exp(log_sigma_decoder)) ** 2, axis = 0)
        else:
            logpxz = np.sum(x * np.log(y) + (1 - x) * np.log(1 - y), axis = 0)

        KLD = 0.5 * np.sum(1 + 2 * log_sigma_encoder -
                           mu_encoder ** 2 - np.exp(2 * log_sigma_encoder), axis = 0)
        lowerbound_x = logpxz + KLD        
        
        # then compute the weights
        log_ws_matrix = lowerbound_x.reshape(K, N)
        # now compute the IWAE bound
        lowerbound = np.sum(logsumexp(log_ws_matrix, axis = 0) - np.log(K))
        
        return lowerbound

    def _computeGradients(self, x, K = 50):
        """Perform backpropagation

        Parameters
        ----------

        x: array-like, shape (n_features, batch_size)
            The data to use for computing gradients to update the weights and biases

        Returns
        -------
        gradients : dictionary with ten or twelve array-like gradients to the weights and biases
        lowerbound : int
            Lower bound on the log likelihood per data point
        """
        # define network
        
        W1, W2, W3, W4, W5 = self.params["W1"], self.params["W2"], self.params["W3"],\
        self.params["W4"], self.params["W5"]
        b1, b2, b3, b4, b5 = self.params["b1"], self.params["b2"], self.params["b3"],\
        self.params["b4"], self.params["b5"]

        if self.continuous:
            W6, b6 = self.params["W6"], self.params["b6"]
            activation = lambda x: np.log(1 + np.exp(x))
        else:
            activation = lambda x: np.tanh(x,x)

        sigmoid = lambda x: 1. / (1 + np.exp(-x))        
        
        # IWAE: first replicate the input
        N = x.shape[1]
        #x = np.repeat(x, K, axis = 1)
        #x = np.tile(x.reshape(-1, 1), (1, K)).reshape(x.shape[0], K * N)       
        x = np.tile(x, (1, K))

        # compute forward pass
        h_encoder = activation(W1.dot(x) + b1)

        mu_encoder = W2.dot(h_encoder) + b2
        log_sigma_encoder = 0.5 * (W3.dot(h_encoder) + b3)


        eps = np.random.randn(self.n_hidden_variables, x.shape[1])
        z = mu_encoder + np.exp(log_sigma_encoder) * eps

        h_decoder = activation(W4.dot(z) + b4)

        y = sigmoid(W5.dot(h_decoder) + b5)

        if self.continuous:
            log_sigma_decoder = 0.5 * (W6.dot(h_decoder) + b6)
            logpxz = np.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_decoder) -
                            0.5 * ((x - y) / np.exp(log_sigma_decoder)) ** 2)
        else:
            logpxz = np.sum(x * np.log(y) + (1 - x) * np.log(1 - y))

        KLD = 0.5 * np.sum(1 + 2 * log_sigma_encoder -
                           mu_encoder ** 2 - np.exp(2 * log_sigma_encoder))
        lowerbound = (logpxz + KLD) / K

        # Compute gradients
        if self.continuous:
            dp_dy = (np.exp(-2 * log_sigma_decoder) * 2 * (x - y)) / 2
            dp_dlogsigd = np.exp(-2 * log_sigma_decoder) * (x - y) ** 2 - 1
        else:
            dp_dy = (x / y - (1 - x) / (1 - y))

        # W5
        dy_dSig = (y * (1 - y))
        dp_dW5 = (dp_dy * dy_dSig).dot(h_decoder.T)
        dp_db5 = np.sum(dp_dy * dy_dSig, axis=1)[:, np.newaxis]

        if self.continuous:
            # W6
            dp_dW6 = (dp_dlogsigd * dy_dSig).dot(0.5 * h_decoder.T)
            dp_db6 = np.sum(dp_dlogsigd * dy_dSig, axis=1)[:, np.newaxis]

        dSig_dHd = W5
        dp_dHd = ((dp_dy * dy_dSig).T.dot(dSig_dHd)).T

        if self.continuous:
            dHd_df = np.exp(W4.dot(z) + b4) / (np.exp(W4.dot(z) + b4) + 1)
        else:
            dHd_df = 1 - h_decoder ** 2

        # W4
        dp_dW4 = (dp_dHd * dHd_df).dot(z.T)
        dp_db4 = np.sum(dp_dHd * dHd_df, axis=1)[:, np.newaxis]

        dtanh_dz = W4
        dmue_dW2 = h_encoder

        dp_dz = (dp_dHd * dHd_df).T.dot(dtanh_dz)
        dp_dmue = dp_dz.T

        dp_dW2 = (dp_dmue).dot(dmue_dW2.T)
        dp_db2 = dp_dmue

        dKLD_dmue = -mu_encoder
        dKLD_dW2 = (dKLD_dmue).dot(dmue_dW2.T)
        dKLD_db2 = dKLD_dmue

        # W2
        dp_dW2 += dKLD_dW2
        dp_db2 = np.sum(dp_db2 + dKLD_db2, axis=1)[:, np.newaxis]

        dz_dlogsige = eps * np.exp(log_sigma_encoder)
        dp_dlogsige = dp_dz.T * dz_dlogsige

        dlogsige_dW3 = 0.5 * h_encoder
        dlogsige_db3 = 0.5

        dp_dW3 = (dp_dlogsige).dot(dlogsige_dW3.T)
        dp_db3 = dp_dlogsige * dlogsige_db3

        dKLD_dlogsige = 1 - np.exp(2 * log_sigma_encoder)
        dKLD_dW3 = (dKLD_dlogsige).dot(dlogsige_dW3.T)
        dKLD_db3 = dKLD_dlogsige * dlogsige_db3

        # W3
        dp_dW3 += dKLD_dW3
        dp_db3 = np.sum(dp_db3 + dKLD_db3, axis=1)[:, np.newaxis]

        # W1, log p(x|z)
        ###########################################
        dmue_dHe = W2
        if self.continuous:
            dHe_df = np.exp(W1.dot(x) + b1) / (np.exp(W1.dot(x) + b1) + 1)
        else:
            dHe_df = 1 - h_encoder ** 2

        dtanh_dW1 = x

        # W1: log(P(x|z)), mu encoder side
        dp_dHe = dp_dmue.T.dot(dmue_dHe)
        dp_dtanh = dp_dHe.T * dHe_df
        dp_dW1_1 = (dp_dtanh).dot(dtanh_dW1.T)
        dp_db1_1 = dp_dtanh

        # W1: log(P(x|z)), log sigma encoder side
        dlogsige_dHe = 0.5 * W3
        dp_dHe_2 = dp_dlogsige.T.dot(dlogsige_dHe)

        dp_dtanh_2 = dp_dHe_2.T * dHe_df
        dp_dW1_2 = (dp_dtanh_2).dot(dtanh_dW1.T)
        dp_db1_2 = dp_dtanh_2

        dp_dW1 = dp_dW1_1 + dp_dW1_2
        dp_db1 = dp_db1_1 + dp_db1_2
        ##########################################

        #W1, DKL
        ###########################################
        dKLD_dHe_1 = dKLD_dlogsige.T.dot(dlogsige_dHe)
        dKLD_dHe_2 = dKLD_dmue.T.dot(dmue_dHe)

        dKLD_dtanh = dKLD_dHe_1.T * dHe_df
        dKLD_dW1_1 = (dKLD_dtanh).dot(dtanh_dW1.T)
        dKLD_db1_1 = dKLD_dtanh

        dKLD_dtanh_2 = dKLD_dHe_2.T * dHe_df
        dKLD_dW1_2 = (dKLD_dtanh_2).dot(dtanh_dW1.T)
        dKLD_db1_2 = dKLD_dtanh_2

        dKLD_dW1 = dKLD_dW1_1 + dKLD_dW1_2
        dKLD_db1 = dKLD_db1_1 + dKLD_db1_2
        ############################################

        # W1
        dp_dW1 += dKLD_dW1
        dp_db1 = np.sum(dp_db1 + dKLD_db1, axis=1)[:, np.newaxis]

        gradients = {"W1": dp_dW1, "W2": dp_dW2, "W3": dp_dW3, "W4": dp_dW4, "W5": dp_dW5,
                     "b1": dp_db1, "b2": dp_db2, "b3": dp_db3, "b4": dp_db4, "b5": dp_db5}

        if self.continuous:
            gradients.update({"W6": dp_dW6, "b6": dp_db6})

        return gradients, lowerbound

    def _updateParams(self, minibatch, N, K):
        """Perform one update on the parameters

        Parameters
        ----------
        minibatch : array-like, shape (n_features, batch_size)
            The data to use for computing gradients to update the weights and biases
        N : int
            The total number of datapoints, used for prior correction


        Returns
        -------
        lowerbound : int
            Lower bound on the log likelihood per data point

        """
        total_gradients, lowerbound = self._computeGradients(minibatch.T, K)

        self.t += 1
        for key in self.params:
            if "W" in key:
                prior = self.params[key]
            else:
                prior = 0
            gradients = total_gradients[key] - prior * (minibatch.shape[0] / N)

            self.gradMean[key] += (1 - self.betaMean) * \
                                      (gradients - self.gradMean[key])
            self.gradVar[key] += (1 - self.betaVar) * \
                                     (gradients ** 2 - self.gradVar[key])            
            alpha = self.learning_rate * np.sqrt(1 - self.betaVar ** self.t) \
                        / (1 - self.betaMean ** self.t)
            epsi = self.epsi * np.sqrt(1 - self.betaVar ** self.t)
            self.params[key] += alpha * self.gradMean[key] \
                               / (np.sqrt(self.gradVar[key]) + epsi)

        return lowerbound

    def fit(self, X, K = 50):
        """Fit SGVB to the data

        Parameters
        ----------
        X : array-like, shape (N, n_features)
            The data that the SGVB needs to fit on

        Returns
        -------
        list_lowerbound : list of int
        list of lowerbound over time
        """
        [N, dimX] = X.shape
        if not self.initialized:
            self._initParams(dimX)
        list_lowerbound = np.array([])
        list_time = np.array([])

        batches = np.arange(0, N, self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches, N)

        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            iteration_lowerbound = 0

            for j in xrange(0, len(batches) - 2):
                minibatch = X[batches[j]:batches[j + 1]]
                lowerbound = self._updateParams(minibatch, N, K)
                iteration_lowerbound += lowerbound

            end = time.time()
            if self.verbose:                
                print("[%s] Iteration %d, lower bound = %.2f,"
                      " time = %.2fs"
                      % (self.__class__.__name__, iteration,
                         iteration_lowerbound / N, end - begin))
            
            list_time = np.append(list_time, end - begin)
            begin = end

            list_lowerbound = np.append(
                list_lowerbound, iteration_lowerbound / N)
        return list_lowerbound, list_time

    def transform(self, X):
        """Transform the data

        Parameters
        ----------
        X : array-like, shape (N, n_features)
            The data that needs to be transformed

        Returns
        -------
        X : array-like, shape (N, n_components_decoder)
            The transformed data
        """
        if self.continuous:
            return np.log(1 + np.exp(X.dot(self.params["W1"].T) + self.params["b1"].T))
        else:
            return np.tanh(X.dot(self.params["W1"].T) + self.params["b1"].T)

    def fit_transform(self, X):
        """Fit and transform the data, wrapper for fit and transform

        Parameters
        ----------
        X : array-like, shape (N, n_features)
            The data that needs to be fitted to and transformed

        Returns
        -------
        X : array-like, shape (N, n_components_decoder)
            The transformed data
        """
        self.fit(X)
        return self.transform(X)

    def score(self, X, K = 1):
        """Computer lower bound on data, very naive implementation

        Parameters
        ----------
        X : array-like, shape (N, n_features)
            The data that needs to be fitted to and transformed

        Returns
        -------
        lower bound : int
            The lower bound on the log likelihood 
        """
        lowerbound = 0.0
        batch_size = 100
        num_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        for j in xrange(num_batches):
            indl = j * batch_size
            indr = min(X.shape[0], (j+1) * batch_size)
            minibatch = X[indl:indr]
            lowerbound += self._lowerbound(minibatch.T, K)
        
        return lowerbound/X.shape[0]
