"""
Variational Autoencoder with alpha-divergence.
"""

import sys
sys.path.append('../')
import aae_reparam as AlphaAutoencoder
import numpy as np
import cPickle
import argparse

def load_data(ratio = 0.9, seed = 0):
    # load and split data
    print "Loading data"
    f = open('freyfaces.pkl','rb')
    data = cPickle.load(f)
    f.close()
    
    np.random.seed(seed)
    np.random.shuffle(data)
    num_train = int(ratio * data.shape[0])
    data_train = data[:num_train]
    data_test = data[num_train:]
    
    return data_train, data_test
    
def main(dimZ, hidden_layers, n_iters, alpha = 1.0, num_samples = 1, \
        opt_method = 'ADAM', decay = 1.0, save = False, seed = 0, backward_pass = 'full'):
    # train AAE for unsupervised learning
    # configurations
    L = 1
    batch_size = 100
    continuous = True
    verbose = True
    learning_rate = 0.0005
    
    # load data
    ratio = 0.9

    data_train, data_test = load_data(ratio, seed)
    
    # initialise the autoencoder
    variables_size = [data_train.shape[1], dimZ]
    encoder = AlphaAutoencoder.VA(variables_size, hidden_layers, \
                  learning_rate, batch_size, continuous, verbose, alpha)
              
    print "Training..."
    num_iter_trained = 0
    for n_iter in n_iters:
        lowerbound = encoder.fit(data_train, num_samples, n_iter, \
            opt_method = opt_method, backward_pass = backward_pass)
        num_iter_trained += n_iter
        print "Evaluating test data..."
        alpha_test = 0.0
        lowerbound_test, time_test = encoder.score(data_test, \
            alpha = alpha_test, num_samples = 100)
        if alpha_test == 0.0:
            print "test data log-likelihood = %.2f, time = %.2fs, iter %d" \
                % (lowerbound_test, time_test, num_iter_trained)
        if alpha_test == 1.0:
            print "test data VI lowerbound = %.2f, time = %.2fs, iter %d" \
                % (lowerbound_test, time_test, num_iter_trained)
        if decay < 1.0:
            encoder.decay_alpha(decay)

    # save model
    if save:
        import pickle
        dataset = 'freyface'
        fname = dataset + '_alpha%.2f_decay%.fseed%d.pkl' \
            % (alpha, decay, seed)
        f = open(fname, 'wb')
        model = encoder.get_model()
        pickle.dump(model, f)
        print 'model saved in file', fname

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AAE experiments.')
    parser.add_argument('--num_layers', '-l', type=int, choices=[1, 2], default=1)
    parser.add_argument('--num_samples', '-k', type=int, default=1)
    parser.add_argument('--alpha', '-a', type=float, default=1.0)
    parser.add_argument('--dimZ', '-Z', type=int, default=5)
    parser.add_argument('--dimH', '-H', type=int, default=200)
    parser.add_argument('--iter', '-i', type=int, default=100)
    parser.add_argument('--opt_method', '-o', type=str, default='ADAM')
    parser.add_argument('--decay', '-d', type=float, default=1.0)
    parser.add_argument('--save_model', '-s', action='store_true', default=False)
    parser.add_argument('--seed', '-S', type=int, default=0)
    parser.add_argument('--backward_pass', '-b', type=str, default='full')
    
    args = parser.parse_args()
    dimZ = args.dimZ
    num_hidden_unit = args.dimH
    num_layers = args.num_layers
    hidden_layers = [[num_hidden_unit for i in xrange(num_layers)]]
    alpha = args.alpha	# default: VI
    alpha = min(alpha, 1.0)
    #alpha = max(0.0, alpha)
    num_samples = args.num_samples
    num_samples = max(num_samples, 1)
    opt_method = args.opt_method
    decay = args.decay
    decay = min(1.0, decay)
    decay = max(0.0, decay)
    num_iters = args.iter
    seed = args.seed    
    if args.backward_pass not in ['full', 'single', 'max']:
        args.backward_pass = 'full'
    
    print 'settings:'
    print 'alpha:', alpha
    print 'dimZ:', dimZ
    print 'hidden layer sizes:', hidden_layers
    print 'num. samples:', num_samples
    print 'optimization method:', opt_method
    print 'alpha decay ratio:', decay
    print 'seed:', seed 
    print 'backward pass method:', args.backward_pass
    
    iter_each_round = 10
    num_rounds = num_iters / iter_each_round
    n_iters = list(np.ones(num_rounds, dtype = int) * iter_each_round)
    main(dimZ, hidden_layers, n_iters, alpha, num_samples, opt_method, 
        decay, args.save_model, seed, args.backward_pass)

