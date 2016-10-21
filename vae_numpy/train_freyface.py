"""
This implementation is based on y0ast code (2014 version)
https://github.com/y0ast/Variational-Autoencoder/tree/5c06a7f14de7f872d837cd4268ee2d081a90056d
"""

#import vrmax as VariationalAutoencoder
#import iwae as VariationalAutoencoder
import vae as VariationalAutoencoder
import numpy as np
import cPickle

print "Loading data"
f = open('freyfaces.pkl','rb')
data = cPickle.load(f)
f.close()


[N,dimX] = data.shape
HU_decoder = 200
HU_encoder = 200

dimZ = 20
L = 1
learning_rate = 0.0005

batch_size = 100
continuous = True
n_iter = 1
verbose = True

encoder = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimZ,\
    learning_rate,batch_size,n_iter,L,continuous,verbose)

print "Iterating"
np.random.seed(0)
np.random.shuffle(data)
num_train = int(data.shape[0] * 0.9)
data_train = data[:num_train]
data_test = data[num_train:]

test_ll_list = []
time_list = []
K = 50	# num of samples

for i in xrange(5):
    lowerbound, time = encoder.fit(data_train, K = K)
    # now evaluate the test ll
    test_ll = encoder.score(data_test, K = 10)
    print 'test-ll:', test_ll, 'avg. time:', time.mean()
    test_ll_list.append(test_ll)
    time_list.append(time.mean())

# save results
f = open('results_vae.pkl', 'wb')
cPickle.dump([test_ll_list, time_list], f)


