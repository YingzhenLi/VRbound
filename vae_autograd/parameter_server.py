# using autograd to do gradient computations
# need autograd package
# author: Yingzhen Li

import autograd.numpy as np

class Parameter_Server:
    """
    Storing all the parameters for the network
    """
    def __init__(self, opt_method = 'ADAM', add_noise = False):
        """
        Initialise the storage of parameters
        """
        self.opt_method = opt_method
        self.layers = dict()
        self.gradVar = dict()
        if self.opt_method in ['ADAM', 'ADADELTA']:
            self.gradMean = dict()
        self.add_noise = add_noise
        if self.add_noise:
            self.noise_eta = 0.01
            self.noise_gamma = 0.55
    
    def add_params(self, params, layer):
        """
        Create all weight and bias parameters with the right dimensions.
        Note: prefix = 'qk' or prefix = 'pk', for the kth layer.
        """
        sigmaInit = 0.01
        prefix = layer.prefix
        layer_sizes = layer.layer_sizes
        prob_type = layer._prob_type
        W_prefix = prefix + 'W'
        b_prefix = prefix + 'b'
        num_layers = len(layer_sizes) - 1
        param_names = []        
        
        # for intermediate layers
        for i in xrange(num_layers - 1):
            W = np.random.normal(0, sigmaInit, (layer_sizes[i+1], layer_sizes[i]))
            b = np.random.normal(0, sigmaInit, (layer_sizes[i+1], 1))
            params[(W_prefix + str(i))] = W
            params[(b_prefix + str(i))] = b
            param_names.extend([(W_prefix + str(i)), (b_prefix + str(i))])
            
        # for the outputs
        W = np.random.normal(0, sigmaInit, (layer_sizes[-1], layer_sizes[-2]))
        b = np.random.normal(0, sigmaInit, (layer_sizes[-1], 1))
        params[(prefix + 'Wmu')] = W
        params[(prefix + 'bmu')] = b
        param_names.extend([prefix + 'Wmu', prefix + 'bmu'])
        
        if prob_type == 'gaussian':
            W = np.random.normal(0, sigmaInit, (layer_sizes[-1], layer_sizes[-2]))
            b = np.random.normal(0, sigmaInit, (layer_sizes[-1], 1))
            params[(prefix + 'WlogSig')] = W
            params[(prefix + 'blogSig')] = b
            param_names.extend([prefix + 'WlogSig', prefix + 'blogSig'])
            
        self.layers[prefix] = param_names
        
        return params
        
    def init_gradient_storage(self, params):
        """
        Initialise the space to store gradient info.
        Here we assume that all parameters have been initialised.
        And we use ADAM or ADAGRAD optimiser.
        """
        self.t = 0
        if self.opt_method == 'ADAM':
            self.betaMean = 0.9
            self.betaVar = 0.999
            self.epsi = 10e-8 
            
        if self.opt_method == 'ADADELTA':
            self.betaMean = 0.9
            self.betaVar = 0.9
            self.epsi = 10e-6       
        
        for key in params:
            self.gradVar[key] = params[key] * 0.0
            if self.opt_method in ['ADAM', 'ADADELTA']:
                self.gradMean[key] = params[key] * 0.0
        
    def update(self, params, gradients, learning_rate, M, N, update_params = True):
        """
        Update the network parameters given the gradients.
        """
        self.t += 1
           
        for key in params:
            # compute the gradient wrt. prior
            if "W" in key:
                prior = 0.5 * params[key]
                gradients[key] -= prior * (M / N)
                
            if self.add_noise:
                sigma_noise = np.sqrt(self.noise_eta / \
                    (1 + self.t) ** self.noise_gamma)
                shape = gradients[key].shape
                gradients[key] += np.random.normal(size=shape) * sigma_noise
                
            # store gradient info
            if self.opt_method == 'ADAM':
                self.gradMean[key] += (1 - self.betaMean) * \
                                      (gradients[key] - self.gradMean[key])
                self.gradVar[key] += (1 - self.betaVar) * \
                                     (gradients[key] ** 2 - self.gradVar[key])
            if self.opt_method == 'ADAGRAD':
                self.gradVar[key] += gradients[key] ** 2
            if self.opt_method == 'ADADELTA':
                self.gradVar[key] += (1 - self.betaVar) * \
                                     (gradients[key] ** 2 - self.gradVar[key])       
                update = gradients[key] * \
                    np.sqrt(self.gradMean[key] + self.epsi) / \
                    np.sqrt(self.gradVar[key] + self.epsi)
                # here gradMean in fact stores the difference between updates
                self.gradMean[key] += (1 - self.betaMean) * \
                                      (update ** 2 - self.gradMean[key])
            
            if update_params:
                if self.opt_method == 'ADAM':
                    alpha = learning_rate * np.sqrt(1 - self.betaVar ** self.t) \
                            / (1 - self.betaMean ** self.t)
                    epsi = self.epsi * np.sqrt(1 - self.betaVar ** self.t)
                    params[key] += alpha * self.gradMean[key] \
                                   / (np.sqrt(self.gradVar[key]) + epsi)
                if self.opt_method == 'ADAGRAD':
                    params[key] += learning_rate / np.sqrt(self.gradVar[key]) * gradients[key]
                if self.opt_method == 'ADADELTA':
                    params[key] += update
                
        return params
            
