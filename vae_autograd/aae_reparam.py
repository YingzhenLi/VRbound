# using autograd to do gradient computations
# need autograd package
# author: Yingzhen Li

import time
import autograd.numpy as np
from autograd import value_and_grad
from autograd.scipy.misc import logsumexp
from network import Network
from parameter_server import Parameter_Server

class VA:
    """
    An auto-encoder with variational Bayes inference.
    """

    def __init__(self, variables_size, hidden_layers, learning_rate=0.01, \
                 batch_size=100, continuous=False, verbose=False, alpha = 1.0):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self._param_initialised = False
        # test: if alpha !=0, we're doing the alpha-SGVB
        # with "tied" two recognition model r and q
        # TODO: let's try alpha < 0!
        #if alpha < 10e-5:
        #    alpha = 0.0		# Doing IWAE
        if alpha > 1 - 10e-5:
            alpha = 1.0		# Doing normal SGVB
        self.alpha = alpha
        print 'alpha =', self.alpha
        if self.alpha < 0.0:
            print 'WARNING: I am not sure if the sampling estimation',
            print 'is still a lower bound!'

        # define encoder network q
        self.qNet = Network('q', variables_size, hidden_layers, \
                 activation = 'log_sigmoid', prob_type = 'gaussian')
                 
        # define decoder network p
        variables_size = list(reversed(variables_size))
        hidden_layers = list(list(reversed(l)) for l in hidden_layers)
        hidden_layers = list(reversed(hidden_layers))
        self.pNet = Network('p', variables_size, hidden_layers, \
                 activation = 'log_sigmoid', prob_type = 'gaussian')
                 
    def _set_alpha(self, alpha):
        # reset value of alpha
        self.alpha = alpha
        
    def decay_alpha(self, decay):
        """
        Do (exponential) decay on alpha
        """     
        self.alpha = self.alpha * decay
        print 'alpha decayed to', self.alpha

    def _initParams(self, opt_method):
        """
        Create all weight and bias parameters with the right dimensions.
        """
        self.params = dict()
        self.param_server = Parameter_Server(opt_method)
        
        # add parameters
        for layer in self.pNet.layers:
            self.params = self.param_server.add_params(self.params, layer)
        for layer in self.qNet.layers:
            self.params = self.param_server.add_params(self.params, layer)
            
        # initialise the storage for gradient info
        self.param_server.init_gradient_storage(self.params)
        self._param_initialised = True
        
    def _initGradientInfo(self, minibatch, N):
        """
        Initialize the gradient info for ADAGRAD
        """
        gradients, lowerbound, bound_type = self._computeGradients(minibatch.T)
        self.param_server.update(self.params, gradients, self.learning_rate, \
                                 minibatch.shape[0], N, update_params = False)
    
    def _comp_log_weights(self, params, x, num_samples = 1):
        """
        Compute log p(theta, D) / q(theta)
        """
        # compute log_q
        z, logq = self.qNet.encode_and_log_prob(params, \
            self.param_server, x, num_samples)
        # compute log_p
        x_tmp = np.repeat(x, num_samples, axis = 1)
        _, logpxz = self.pNet.encode_and_log_prob(params, \
            self.param_server, z, data = x_tmp)        
        logp0 = np.sum(-(0.5 * np.log(2 * np.pi)) - 0.5 * z ** 2, axis = 0)
        # compute log importance weights
        logF = logp0 + logpxz - logq
        
        return logF
        
    def _lowerbound(self, params, x, num_samples = 1, alpha = 1.0):
        """
        Compute the alpha-VI objective
        """        
        # compute lowerbound
        logF = self._comp_log_weights(params, x, num_samples)
        batchsize = x.shape[1]
        lowerbound = 0.0
        if alpha < 1 - 10e-10:
            # compute the normaliser for each q(z|x_n)
            logF = (1 - alpha) * logF
            for i in xrange(batchsize):
                indl = int(i * num_samples); indr = int((i+1) * num_samples)
                lowerbound = lowerbound + logsumexp(logF[indl:indr]) \
                    - np.log(num_samples)            
            lowerbound = lowerbound / (1 - alpha)
        else:
            # VI bound
            for i in xrange(batchsize):
                indl = int(i * num_samples); indr = int((i+1) * num_samples)
                lowerbound = lowerbound + np.sum(logF[indl:indr]) / num_samples

        return lowerbound
        
    def _backprop_single(self, params, x, num_samples = 1, alpha = 1.0):
        """
        Efficient training by computing k forward pass and only 1
        backward pass (by sampling particles according to the weights).
        For VI all the weights are equal.
        """
        # compute weights
        logF = self._comp_log_weights(params, x, num_samples)
        batchsize = x.shape[1]
        lowerbound = 0.0
        logFa = (1 - alpha) * logF
        for i in xrange(batchsize):
            indl = int(i * num_samples); indr = int((i+1) * num_samples)
            log_weights = logFa[indl:indr] - logsumexp(logFa[indl:indr])
            prob = list(np.exp(log_weights))
            # current autograd doesn't support np.random.choice!
            sample_uniform = np.random.random()
            for j in xrange(num_samples):
                sample_uniform = sample_uniform - prob[j]
                if sample_uniform <= 0.0:
                    break
            ind_current = indl + j                
            lowerbound = lowerbound + logF[ind_current]

        return lowerbound
        
    def _backprop_max(self, params, x, num_samples = 1, alpha = 1.0):
        """
        When alpha goes to minus infinity, then the importance weights will
        be centering at the mode. So here I try back-propagating the particle 
        with largest (unnormalised) weight.
        """
        
        # compute weights
        logF = self._comp_log_weights(params, x, num_samples)
        batchsize = x.shape[1]        
        
        logF_matrix = logF.reshape(batchsize, num_samples)
        logF_max = np.max(logF_matrix, axis = 1)
        lowerbound = np.sum(logF_max)
        
        return lowerbound
        
    def _loss_grad(self, backward_pass = 'single'):            
        # first determin bound type
        if backward_pass == 'max':
            bound_type = 'max weight particle estimate'
        elif self.alpha > 1 - 10e-7:
            bound_type = 'VI lowerbound'           
        elif self.alpha <= 10e-7 and self.alpha >= 0.0:
            bound_type = 'IS estimate of log marginal'
        else:
            if self.alpha > 0:
                bound_type = 'alpha-VI lowerbound'
            else:
                bound_type = '(negative)-alpha-VI estimate'
            
        # then determine the training type
        if backward_pass == 'max':
            loss_and_grad = value_and_grad(self._backprop_max)
        elif backward_pass == 'full':
            loss_and_grad = value_and_grad(self._lowerbound)
        else:
            loss_and_grad = value_and_grad(self._backprop_single)
            
        return loss_and_grad, bound_type
    
    def _computeGradients(self, minibatch, num_samples = 1, \
            backward_pass = 'single'):
        loss_and_grad, bound_type = self._loss_grad(backward_pass)
        (lowerbound, gradients) = loss_and_grad(self.params, \
                                      minibatch, num_samples, self.alpha)
        return gradients, lowerbound, bound_type

    def _updateParams(self, minibatch, N, num_samples = 3, backward_pass = 'full'):
        """
        Perform one update on the parameters
        """
        
        if backward_pass == 'single' and self.alpha > 1 - 10e-10:
            num_samples = 1
        
        # first get gradients
        gradients, lowerbound, bound_type = \
            self._computeGradients(minibatch.T, num_samples, backward_pass)
        
        # then do updates!
        self.param_server.update(self.params, gradients, \
            self.learning_rate, minibatch.shape[0], N)

        return lowerbound, bound_type

    def fit(self, X, num_samples = 1, n_iter = 100, opt_method = 'ADAM', backward_pass = 'full'):
        """
        Fit alpha-SGVB to the data
        """    
        [N, dimX] = X.shape        
        list_lowerbound = []
        batches = np.arange(0, N, self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches, N)

        if not self._param_initialised:            
            self._initParams(opt_method)
            if self.verbose:
                print "Initialize gradient information storage..."
            for i in xrange(min(5, len(batches)-1)):
                minibatch = X[batches[i]:batches[i + 1]]
                self._initGradientInfo(minibatch, N)

        begin = time.time()
        for iteration in xrange(1, n_iter + 1):
            iteration_lowerbound = 0
            np.random.shuffle(X)

            for j in xrange(0, len(batches) - 2):
                minibatch = X[batches[j]:batches[j + 1]]
                lowerbound, bound_type = \
                    self._updateParams(minibatch, N, num_samples, backward_pass)
                iteration_lowerbound += lowerbound

            if self.verbose:
                end = time.time()
                print("[%s] Iteration %d, %s = %.2f, time = %.2fs"
                      % (self.__class__.__name__, iteration, bound_type,
                         iteration_lowerbound / N, end - begin))
                begin = end

            list_lowerbound.append(iteration_lowerbound / N)
        
        return np.array(list_lowerbound)

    def score(self, X, alpha = 0.0, num_samples = 100):
        """
        Computer lower bound on data, following the IWAE paper.
        """
        begin = time.time()
        print 'num. samples for eval:', num_samples
        lowerbound = self._lowerbound(self.params, \
                                    X.T, num_samples, alpha = alpha)
        end = time.time()
        time_test = end - begin
        lowerbound = lowerbound / X.shape[0]

        return lowerbound, time_test
    
    def get_model(self):
        model = {'alpha': self.alpha, 'params': None}
        if self._param_initialised:
            model['params'] = self.params
    
        return model
        
    def set_model(self, model):
        self.alpha = model['alpha']
        
        self._initParams('SGD')
        self.params = model['params'] 
        if self.params is None:
            self._param_initialised = False
        
