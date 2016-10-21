import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import bivariate_normal
import sys

def comp_true_posterior(mu0, Lbd0, X, y, sig_noise):
	"""
	Compute the true posterior
	"""
	Lbd = Lbd0
	eta = np.dot(mu0, Lbd0)
	N = X.shape[0]
	Lbd = Lbd0 + np.dot(X.T, X) / sig_noise ** 2
	for n in xrange(N):	    
	    eta += y[n] * X[n] / sig_noise ** 2
	mu = np.dot(eta, np.linalg.inv(Lbd))
	
	return mu, Lbd

def comp_log_marginal(mu0, Lbd0, mu, Lbd, y, sig_noise):
    """
    Compute log p(D)
    """
    logprob = - y.shape[0] / 2.0 * np.log(2.0 * np.pi) * sig_noise
    logprob += (np.log(np.linalg.det(Lbd0)) - np.log(np.linalg.det(Lbd))) / 2.0
    logprob += (np.dot(mu, Lbd) * mu).sum() / 2.0
    logprob -= (np.dot(mu0, Lbd0) * mu0).sum() / 2.0
    logprob -= (y ** 2).sum() / 2.0 / sig_noise ** 2
    
    return logprob
    
def comp_divergence(mu1, Lbd1, mu2, Lbd2, alpha):
    if alpha == 0:
        divergence = 0.0
    elif alpha == 1:
        # TODO: KL
        divergence = np.log(np.linalg.det(Lbd1)) - np.log(np.linalg.det(Lbd2))
        mu_diff = mu2 - mu1
        divergence += (np.dot(mu_diff, Lbd2) * mu_diff).sum()
        divergence += np.trace(np.dot(Lbd2, np.linalg.inv(Lbd1))) - Lbd1.shape[0]
    else:
        Lbd = alpha * Lbd1 + (1 - alpha) * Lbd2
        mu = alpha * np.dot(mu1, Lbd1) + (1 - alpha) * np.dot(mu2, Lbd2)
        mu = np.dot(mu, np.linalg.inv(Lbd))
        divergence = alpha * np.log(np.linalg.det(Lbd1)) \
            + (1 - alpha) * np.log(np.linalg.det(Lbd2)) \
            - np.log(np.linalg.det(Lbd))
        divergence += (np.dot(mu, Lbd) * mu).sum()
        divergence -= alpha * (np.dot(mu1, Lbd1) * mu1).sum()
        divergence -= (1 - alpha) * (np.dot(mu2, Lbd2) * mu2).sum()
    
    return divergence / 2.0
	
def comp_approx_posterior(mu, Lbd, alpha, return_matrix = False):
    """
    Compute the approximate posterior
    given alpha and the true posterior params
    """
    
    a = Lbd[0, 0]; b = Lbd[0, 1]; c = Lbd[1, 1]
    if alpha == 1.0:
        Lbd_approx = np.array([a, c])
    elif alpha == 0.0:
        Lbd_approx = np.array([a - b ** 2 / c, c - b ** 2 / a])
    elif np.abs(alpha) > 50.0:
        tmp = np.sqrt(np.array([a / c, c / a])) * np.abs(b)
        if alpha < 0.0:
            Lbd_approx = np.array([a, c]) - tmp
        else:
            Lbd_approx = np.array([a, c]) + tmp 
    else:
        r = b ** 2 / a / c
        rho = 1.0 + (np.sqrt(1 - 4 * alpha * (1 - alpha) * r) - 1) / (2.0 * alpha)
        Lbd_approx = np.array([a * rho, c * rho])
    
    if return_matrix:
        Lbd_approx = np.diag(Lbd_approx)
        
    return mu, Lbd_approx

def gen_data(N, mu0, Lbd0, sig_noise, seed = 500):
    Sigma0 = np.linalg.inv(Lbd0)
    np.random.seed(seed)
    theta = np.random.multivariate_normal(mu0, Sigma0)
    X = np.random.random([N, 2]) * 2.0
    y = (X * theta).sum(1) + np.random.randn(N) * sig_noise
    return X, y    
    
def main():
    
    sig_noise = 1.0
    alpha_list = [-1000.0, 0.0, 0.5, 1.0, 1000.0]	# here we are computing L_{1 - alpha/N}
    color=['r', 'g', 'b', 'm', 'c']
    fig, ax = plt.subplots(figsize=(3,3))
    i = 0
    
    # first compute the true posterior
    mu0 = np.zeros(2)
    Lbd0 = np.eye(2)
    X = np.array([[1.0, -1.0], [-1.0, 1.0]])
    y = np.array([0.0, 0.0])
    #X, y = gen_data(10, mu0, Lbd0, sig_noise)
    mu, Lbd = comp_true_posterior(mu0, Lbd0, X, y, sig_noise)
    
    # then compute the approximations!
    Lbd_list = []
    return_matrix = False
    delta = 0.05
    x_con = np.arange(mu[0] - 1.1, mu[0] + 1.1, delta)
    y_con = np.arange(mu[1] - 1.1, mu[1] + 1.1, delta)
    X_con, Y_con = np.meshgrid(x_con, y_con)
    
    # first plot true posterior       
    Sigma = np.linalg.inv(Lbd)
    det_Sigma = np.linalg.det(Sigma)
    levels = [np.exp(-0.5) / (2 * np.pi) / np.sqrt(det_Sigma)]
    Z_con =  bivariate_normal(X_con, Y_con, 
        np.sqrt(Sigma[0, 0]), np.sqrt(Sigma[1, 1]), mu[0], mu[1], Sigma[0, 1])
    cs = plt.contour(X_con, Y_con ,Z_con, levels = levels, colors = 'k', linewidths=3)
    lines = [cs.collections[0]]
    
    i = 0
    for alpha in alpha_list:
        _, Lbd_approx = comp_approx_posterior(mu, Lbd, alpha, return_matrix)
        Lbd_list.append(Lbd_approx)
        
        # plot contour 
        det_Lbd_approx = Lbd_approx[0] * Lbd_approx[1]
        levels = [np.exp(-0.5) / (2 * np.pi) * np.sqrt(det_Lbd_approx)]       
        Z_con =  bivariate_normal(X_con, Y_con, 
            np.sqrt(1.0 / Lbd_approx[0]), np.sqrt(1.0 / Lbd_approx[1]), mu[0], mu[1])
        cs = plt.contour(X_con, Y_con ,Z_con, levels = levels, 
            colors = color[i], linewidths=3)
        i += 1
        lines.append(cs.collections[0])

    
#    labels = ['exact',
#              r'$\alpha \rightarrow -\infty$', 
#              r'$\alpha = 0.0$', 
#              r'$\alpha = 0.5$',
#              r'$\alpha = 1.0$',
#              r'$\alpha \rightarrow +\infty$']
#    plt.legend(lines, labels)
    ax.set_xlim(left=-1.2, right=1.2)
    ax.set_ylim(ymin=-1.2, ymax=1.5)
    plt.xlabel('', fontsize = 30)
    plt.ylabel('', fontsize = 30)
    plt.axis('equal')
    #plt.show()
    plt.savefig('approx.svg', format='svg')

if __name__== '__main__':
    main()

