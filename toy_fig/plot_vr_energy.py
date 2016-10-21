import numpy as np
from matplotlib import pyplot as plt
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
    elif np.abs(alpha) > 50.0:
        divergence = np.log(np.linalg.det(Lbd1)) - np.log(np.linalg.det(Lbd2))
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
        divergence /= (alpha - 1)
    
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
    
    if (Lbd_approx <= 0.0).any():
        print Lbd_approx, alpha
    
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
    
    sig_noise_list = np.linspace(0.01, 2.0, 100)
    alpha_list = [-1000.0, 0.0, 0.5, 1.0, 1000.0]	# here we are computing L_{1 - alpha/N}
    color=['r', 'g', 'b', 'm', 'c']
    plt.figure()
    i = 0
    
    # first compute the true posterior
    mu0 = np.zeros(2)
    Lbd0 = np.eye(2)
    X = np.array([[1.0, -1.0], [-1.0, 1.0]])
    y = np.array([0.0, 0.0])
    #X, y = gen_data(10, mu0, Lbd0, sig_noise)
    
    
    # then compute the approximations!
    
    log_true = []
    log_approx = []
    for sig_noise in sig_noise_list:
        mu, Lbd = comp_true_posterior(mu0, Lbd0, X, y, sig_noise)
        log_posterior = comp_log_marginal(mu0, Lbd0, mu, Lbd, y, sig_noise)
        log_true.append(log_posterior)
        log_approx_sig = []
        for alpha in alpha_list:
            _, Lbd_approx = comp_approx_posterior(mu, Lbd, alpha, True)            
            log_approx_alpha = log_posterior - comp_divergence(mu, Lbd_approx, mu, Lbd, alpha)
            log_approx_sig.append(log_approx_alpha)
        log_approx.append(log_approx_sig)
        
    # now plot the energy
    log_true = np.array(log_true)
    log_approx = np.array(log_approx)
    
    # first plot the true evidence
    fig, ax = plt.subplots(figsize=(6,3))
    plt.plot(sig_noise_list, log_true, '-', color='k', 
                label='exact', linewidth=2)
    max_true = np.max(log_true)
    arg_max_true = np.argmax(log_true)
    plt.vlines(sig_noise_list[arg_max_true], 0, max_true,
                linestyles='--',colors='k', linewidth=2)
                
    # then plot the approximations
    labels = [r'$\alpha \rightarrow -\infty$', 
              r'$\alpha = 0.0$', 
              r'$\alpha = 0.5$',
              r'$\alpha = 1.0$',
              r'$\alpha \rightarrow +\infty$']
    for i in xrange(len(alpha_list)):
        plt.plot(sig_noise_list, log_approx[:, i], '-', color=color[i], 
                label=labels[i], linewidth=3)
        max_approx = np.max(log_approx[:, i])
        arg_max_approx = np.argmax(log_approx[:, i])
        plt.vlines(sig_noise_list[arg_max_approx], 0, max_approx,
                linestyles='--',colors=color[i], linewidth=3)
    
    plt.legend(loc='lower left')
    plt.xlabel(r'$\sigma$', fontsize = 25)
    plt.ylabel(r'$\mathcal{L}_{\alpha}$', fontsize = 25)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_ylim(ymin = -8.0, ymax = 0.0)
    plt.savefig('log_evidence.svg', format='svg')
    plt.show()

if __name__== '__main__':
    main()
        
