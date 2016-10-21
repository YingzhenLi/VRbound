# using autograd to do gradient computations
# need autograd package
# author: Yingzhen Li

import autograd.numpy as np

class network_layer(object):
    """
    Network layer Object.
    Note that the intermidiate hidden layers are considered
    as deterministic variables.
    """
    
    def __init__(self, prefix, input_size, output_size, hidden_layers, \
                 activation = 'sigmoid', prob_type = 'gaussian'):
        """
        Initialise the network layer
        """
        self.prefix = prefix	# string, 'qk' or 'pk'
        self.layer_sizes = [input_size]
        self.layer_sizes.extend(hidden_layers)
        self.layer_sizes.append(output_size)
        self.num_layers = len(self.layer_sizes) - 1
        self.hidden_layers = hidden_layers
        # default activation function is sigmoid
        if activation == 'sigmoid':
            self._activation = lambda x: 1.0 / (1.0 + np.exp(-x))
        if activation == 'log_sigmoid':
            self._activation = lambda x: np.log(1 + np.exp(x))
        # default output probability is gaussian
        self._prob_type = prob_type
        
    def _encode_layer(self, inputs, W, b):
        """
        Compute outputs for the lth layers
        """
        # inputs is of shape (input_dim, num_inputs)
        return self._activation(np.dot(W, inputs) + b)
        
    def _encode(self, params, x, num_samples = 1):
        """
        Sampling h from proposal p(h|x)        
        """
        # unpack parameters
        W_prefix = self.prefix + 'W'
        b_prefix = self.prefix + 'b'

        # first compute encodings
        h_out = x
        for i in xrange(self.num_layers - 1):
            W = params[(W_prefix + str(i))]
            b = params[(b_prefix + str(i))]
            h_out = self._encode_layer(h_out, W, b)
            
        Wmu = params[(self.prefix + 'Wmu')]
        bmu = params[(self.prefix + 'bmu')]     
        mu_out = np.dot(Wmu, h_out) + bmu
        if 'p' in self.prefix:
            sigmoid = lambda x: 1. / (1 + np.exp(-x))
            mu_out = sigmoid(mu_out)
        
        if self._prob_type == 'gaussian':
            Wsig = params[(self.prefix + 'WlogSig')]
            bsig = params[(self.prefix + 'blogSig')]
            log_sigma_out = 0.5 * (np.dot(Wsig, h_out) + bsig)

        # then do samping
        if self._prob_type == 'gaussian':
            mu_out = np.repeat(mu_out, num_samples, axis = 1)
            log_sigma_out = np.repeat(log_sigma_out, num_samples, axis = 1)
            eps = np.random.normal(0, 1, [self.layer_sizes[-1], x.shape[1] * num_samples])
            output = mu_out + np.exp(log_sigma_out) * eps       
            param_out = (mu_out, log_sigma_out)
        
        return output, param_out
        
    def _compute_log_prob(self, output, param_out):
        """
        Compute log p(output|input)
        """
        if self._prob_type == 'gaussian':
            (mu_out, log_sigma_out) = param_out
            logq = -(0.5 * np.log(2 * np.pi) + log_sigma_out) - \
                       0.5 * ((output - mu_out) / np.exp(log_sigma_out)) ** 2
                       
        logq = np.sum(logq, axis = 0)
        
        return logq
        
    def _encode_and_prob(self, params, x, num_samples = 1):
        """
        Compute samples and return the log probability
        """
        output, param_out = self._encode(params, x, num_samples)
        logq = self._compute_log_prob(output, param_out)
        
        return output, logq
        
                
        
