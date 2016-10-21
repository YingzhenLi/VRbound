# using autograd to do gradient computations
# need autograd package
# author: Yingzhen Li

import autograd.numpy as np
from network_layer import network_layer

class Network(object):
    """
    Network for encoder/decoder.
    """
    
    def __init__(self, prefix, variables_size, hidden_layers, \
                 activation = 'log_sigmoid', prob_type = 'gaussian'):
       """
       Initialise the network structure
       """
       self.prefix = prefix 	# 'q' or 'p'
       self.num_layers = len(variables_size) - 1
       self.variables_size = variables_size
       
       # initialise the layers
       self.layers = []
       for l in xrange(self.num_layers):
           prefix_layer = self.prefix + str(l)
           input_size = self.variables_size[l]
           output_size = self.variables_size[l+1]
           hidden_layers_l = hidden_layers[l]
           layer = network_layer(prefix_layer, input_size, output_size, \
                                 hidden_layers_l, activation, prob_type)
           self.layers.append(layer)
           print 'network structure:', layer.layer_sizes
       
       print '%d layers in %s network' % (len(self.layers), self.prefix)
       
        
    def encode_and_log_prob(self, params, param_server, inputs, num_samples = 1, data = None):
        """
        Compute outputs and return the log probability
        """
        l = 0
        output = inputs
        logq = 0.0
        if self.prefix == 'q':
            data = None		# q network
        for layer in self.layers:
            if l == 0:
                num_samples_l = num_samples
            else:
                num_samples_l = 1
            output, param_out = layer._encode(params, output, num_samples_l)
            if (l == self.num_layers - 1) and (data is not None):
                logq = logq + layer._compute_log_prob(data, param_out)
            else:
                logq = logq + layer._compute_log_prob(output, param_out)
            l += 1
        
        return output, logq
    
    
    
