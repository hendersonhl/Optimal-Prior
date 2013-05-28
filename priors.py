"""Source code for priors project

Key notation:
    T: Number of observations
    K: Number of covariates (including intercept)
    M: Parameter support points
    J: Error suppor points
    y: Dependent variable (T-by-1)
    X: Covariate matix (T-by-K)
    q: Prior on signal (KM-by-1)
    u: Prior on noise (TJ-by-1)
    z: Parameter support vector (M-by-1)
    v: Noise support vector (J-by-1)
    lmda: Lagrange multipliers (T-by1)

"""

import numpy as np
from itertools import permutations
from scipy.misc import logsumexp
from scipy.optimize import fmin
import matplotlib.pyplot as plt

def dual(lmda, q, u, y, X, z, v):
    """Returns value of dual objective function.
    
    Parameters
    ----------
    lmda : ndarray
        Lagrange multipliers
    q : ndarray
        Prior on signal
    u : ndarray
        Prior on noise
    y : ndarray
        Dependent variable  
    X : ndarray
        Covariate matrix
    z : ndarray
        Parameter support vector
    v : ndarray  
        Noise support vector  
    
    """
    assert np.shape(y)[1] == 1, 'y dimension issue'
    assert np.shape(q)[1] == 1, 'q dimension issue'
    assert np.shape(u)[1] == 1, 'u dimension issue'
    assert np.shape(z)[1] == 1, 'z dimension issue'
    assert np.shape(v)[1] == 1, 'v dimension issue'
    lmda = lmda[:, np.newaxis]
    T = np.shape(X)[0]
    K = np.shape(X)[1]
    M = np.shape(z)[0]
    J = np.shape(v)[0]
    eye_k = np.eye(K)
    eye_t = np.eye(T)
    Z = np.kron(eye_k, z.T)
    V = np.kron(eye_t, v.T)
    p1 = np.dot(y.T, lmda)[0][0] 
    assert type(p1) == np.float64, 'p1 must be float'
    p2a = np.dot(Z.T, np.dot(X.T, lmda)) # KM-by-1
    p2b = np.reshape(p2a, (-1, M))
    p2c = logsumexp(p2b, axis=1, b=np.reshape(q, (-1, M)))
    p2 = np.sum(p2c)  
    assert type(p2) == np.float64, 'p2 must be float'
    p3a = np.dot(V.T, lmda) # TJ-by-1   
    p3b = np.reshape(p3a, (-1, J))
    p3c = logsumexp(p3b, axis=1, b=np.reshape(u, (-1, J)))
    p3 = np.sum(p3c)
    assert type(p3) == np.float64, 'p3 must be float'
    return p1 - p2 - p3

def pprobs(lmda, y, X, q, z):
    """Returns vector of probabilities on signal.
    
    Parameters
    ----------
    lmda : ndarray
        Lagrange multipliers
    y : ndarray
        Dependent variable  
    X : ndarray
        Covariate matrix
    q : ndarray
        Prior on signal
    z : ndarray
        Parameter support vector 
    
    """
    assert np.shape(y)[1] == 1, 'y dimension issue'
    assert np.shape(q)[1] == 1, 'q dimension issue'
    assert np.shape(z)[1] == 1, 'z dimension issue'
    lmda = lmda[:, np.newaxis]
    K = np.shape(X)[1]
    M = np.shape(z)[0]
    ones_m = np.ones(M)[:, np.newaxis]
    eye_k = np.eye(K)
    Z = np.kron(eye_k, z.T)
    p1a = np.dot(Z.T, np.dot(X.T, lmda))
    p1b = np.exp(p1a)
    p1 = q * p1b
    assert np.shape(p1) == (K*M, 1), 'p1 dimension issue'
    p2a = np.dot(ones_m, ones_m.T)
    p2b = np.kron(eye_k, p2a)
    p2 = np.dot(p2b, p1)
    assert np.shape(p2) == (K*M, 1), 'p2 dimension issue'
    return p1 / p2  
    
def wprobs(lmda, u, v, T):
    """Returns probabilities on noise.
    
    Parameters
    ----------
    lmda : ndarray
        Lagrange multipliers
    u : ndarray
        Prior on noise
    v : ndarray  
        Noise support vector 
    T : int
        Number of observations
        
    """  
    assert np.shape(u)[1] == 1, 'u dimension issue'
    assert np.shape(v)[1] == 1, 'v dimension issue'
    assert type(T) == int, 'T must be integer'
    lmda = lmda[:, np.newaxis]
    J = np.shape(v)[0]
    ones_j = np.ones(J)[:, np.newaxis]
    eye_t = np.eye(T)
    V = np.kron(eye_t, v.T)
    p1a = np.dot(V.T, lmda)
    p1b = np.exp(p1a)
    p1 = u * p1b
    assert np.shape(p1) == (T*J, 1), 'p1 dimension issue'
    p2a = np.dot(ones_j, ones_j.T)
    p2b = np.kron(eye_t, p2a)
    p2 = np.dot(p2b, p1)
    assert np.shape(p2) == (T*J, 1), 'p2 dimension issue'
    return p1 / p2 
    
def coeffs(probs, z):
    """Returns coefficient estimates.
    
    Parameters
    ----------
    probs : ndarray
        Probabilities on signal
    z : ndarray
        Parameter support vector
        
    """
    assert np.shape(probs)[1] == 1, 'probs dimension issue'
    assert np.shape(z)[1] == 1, 'z dimension issue'
    K = len(probs)/len(z)
    eye_k = np.eye(K)
    Z = np.kron(eye_k, z.T)   
    assert np.shape(Z) == (K, K*len(z)), 'Z dimension issue'
    return np.dot(Z, probs)
        
def xdata(T, K, a, b, corr=False):
    """Returns covariate data.

    Parameters
    ----------
    T : int
        Number of observations
    K : int
        Number of covariates (including intercept)
    a : int
        Lower bound on uniform distribution
    b : int
        Upper bound on uniform distribution
    corr : bool
        Correlated covariates

    """
    assert T > 0 and K > 0 and a >= 0 and b > 0, 'inputs must be non-negative'
    assert b > a, 'a must be less than b'
    if not corr: # uncorrelated covariates
        X1 = np.random.uniform(a, b, size=(T, K - 1)) 
        X = np.hstack((np.ones(T)[:, np.newaxis], X1)) # add intercept
    else:  # correlated covariates
        assert K > 2, 'if corr==True, K must be greater than two'
        X1 = np.random.uniform(a, b, size=(T, K - 2)) 
        X2 = 2*X1[:, -1][:, np.newaxis] + np.random.normal(0, 5, size=(T, 1))
        X = np.hstack((np.ones(T)[:, np.newaxis], X1, X2)) 
    assert np.shape(X) == (T, K), 'X dimension issue'
    return X
    
def ydata(X, parms, sd):
    """Returns dependent variable.

    Parameters
    ----------
    X : ndarray
        Covariates
    parms : ndarray
        Model parameters
    sd : int
        Standard deviation on noise
 
    """
    assert sd > 0, 'sd must be greater than zero'
    parms = np.atleast_2d(parms)
    T = np.shape(X)[0] # number of observations
    e = np.random.normal(0, sd, size=(T, 1)) # noise
    assert np.shape(X)[1] == np.shape(parms)[1], 'array shape mismatch'
    y = np.dot(X, parms.T) + e
    assert np.shape(y) == (T, 1), 'y dimension issue'
    return y
    
def ce(pprob, wprob, q, J):
    """Returns cross entropy.

    Parameters
    ----------
    pprobs : ndarray
        Probabilities on signal
    wprobs : ndarray
        Probabilities on noise
    q : ndarray
        Prior on signal 
    J : int
        Number of error support points

    """
    assert np.shape(pprob)[1] == 1, 'pprob dimension issue' 
    assert np.shape(wprob)[1] == 1, 'wprob dimension issue'      
    assert np.shape(q)[1] == 1, 'q dimension issue' 
    assert type(J) == int, 'J must be an integer'  
    u = (1.0/J)*np.ones(len(wprob))[:, np.newaxis] # uniform prior on errors  
    assert len(pprob) == len(q), "len(pprob) not equal to len(q)"  
    assert len(wprob) == len(u), "len(wprob) not equal to len(u)"     
    ce_signal = np.sum(pprob * np.log((pprob + 1e-08)/q))
    ce_noise = np.sum(wprob * np.log((wprob + 1e-08)/u))
    ce_total = ce_signal + ce_noise
    assert type(ce_signal) == np.float64, 'ce_signal must be float'
    assert type(ce_noise) == np.float64, 'ce_noise must be float'
    assert type(ce_total) == np.float64, 'ce_total must be float'
    return ce_signal, ce_noise, ce_total

def priors(M):
    """Yields all possible priors.
  
    Parameters
    ----------
    M : int
        Number of support points
    
    """
    assert M > 0, 'M must be greater than zero'
    assert type(M) == int, 'M must be an integer'
    seq = xrange(1, M + 1)
    prior = []
    for perm in permutations(seq, M):
        normalization = (M * (M + 1)) / 2.
        prior.append(np.asarray(perm)/normalization)
    # uniform prior last
    prior.append(np.ones(M) / M)
    return prior

def esupport(y, M):
    """Returns error support vector.
   
    Parameters
    ----------
    y : ndarray
        Dependent variable
    M : int
        Number of support points
    
    """
    assert np.shape(y)[1] == 1, 'y dimension issue'
    assert type(M) == int, 'M must be an integer'
    assert M==3 or M==5, "M must equal 3 or 5"
    sig = np.std(y)
    if M==3:
        v = np.array([-3*sig, 0, 3*sig])[:, np.newaxis]
    else: # M=5
        v = np.array([-3*sig, -1.5*sig, 0, 1.5*sig, 3*sig])[:, np.newaxis]
    return v
    
def sq_dev(parms, coeffs):
    """Returns sum of squared deviations of estimates from true values.
    
    Parameters
    ----------
    parms : ndarray
        Model parameters
    coeffs : ndarray
        Estimated coefficients
        
    """
    assert type(coeffs) == np.ndarray, 'coeffs must be array'
    assert np.shape(coeffs)[1] == 1, 'coeffs dimension issue'
    assert len(parms) == len(coeffs), 'len(parms) not equal to len(coeffs)'
    parms = np.asarray(parms)[:, np.newaxis]
    sq_dev = np.sum((parms - coeffs)**2)
    assert type(sq_dev) == np.float64, 'sq_dev must be float'
    return sq_dev

def fit_model(obj, priors, x0, y, X, u, z, v, parms):
    """Minimizes chosen objective function. 

    Parameters
    ----------
    obj : callable func
        Objective function
    priors : ndarray
        Possible priors
    x0 : ndarray
        Starting values
    y : ndarray
        Dependent variable
    X : ndarray
        Covariates
    u : ndarray
        Prior on noise
    z : ndarray
        Parameter support vector
    v : ndarray  
        Noise support vector  
    parms : ndarray
        Model parameters       

    """ 
    assert np.shape(y)[1] == 1, 'y dimension issue'  
    assert np.shape(u)[1] == 1, 'u dimension issue'
    assert np.shape(z)[1] == 1, 'z dimension issue'
    assert np.shape(v)[1] == 1, 'v dimension issue'
    T = len(y)
    K = np.shape(X)[1]
    M = len(z)
    J = len(v) 
    q = (1.0/M)*np.ones(K*M)[:, np.newaxis]
    result_dev, result_ce = [], []
    for prior in priors:
        print 'Estimating model with prior: {}'.format(prior)
        # introduce iteration-specific prior into q
        q[M: 2*M] = np.array(prior)[:, np.newaxis]
        assert np.shape(q)[1] == 1, 'q dimension issue' 
        # fit model
        lmda = fmin(obj, x0, args=(q,), ftol=1e-10, disp=0)
        # get results
        pprob = pprobs(lmda, y, X, q, z)
        wprob = wprobs(lmda, u, v, T)       
        ent = ce(pprob, wprob, q, J)
        coeff = coeffs(pprob, z)
        dev = sq_dev(parms, coeff)
        result_dev.append(dev)
        result_ce.append(ent[2])
    return result_dev, result_ce
    

if __name__ == "__main__":

    # set seed
    np.random.seed(123)

    # user inputs
    T = 50 # sample sizes
    nreps = 10 # replications for each sample size
    parms = [1, -5, 2] # parameter values
    a = 0 # lower bound on uniform dist. of covariates
    b = 20 # upper bound on uniform dist. of covariates
    corr = False # correlated covariates
    sd = 2 # standard deviation on model noise
    z = [-25, 0, 25] # support for parameters

    # set-up
    K = len(parms) # number of covariates (including intercept)
    M = J = len(z) # number of support points
    X = xdata(T, K, a, b, corr=corr)
    x0 = np.zeros(T) # starting values
    z = np.array(z)[:, np.newaxis]
    u = (1.0/J)*np.ones(T*J)[:, np.newaxis] # uniform prior on noise
    priors = priors(M)
   
    # experiment
    dev_reps = np.zeros([len(priors)])
    ce_reps = np.zeros([len(priors)])
    for i in range(nreps):
        print 'Enter replication {}'.format(i + 1)
        y = ydata(X, parms, sd)
        v = esupport(y, M)
        # define objective function as maximization of dual
        obj = lambda lmda, q: - dual(lmda, q, u, y, X, z, v) 
        result_dev, result_ce = fit_model(obj, priors, x0, y, X, u, z, v, parms)  
        dev_reps += np.array(result_dev) / nreps   # average outcomes 
        ce_reps += np.array(result_ce) / nreps    
        print 'Exit replication {}'.format(i + 1)

    # plot outcomes
    plt.plot(dev_reps, ce_reps, 'ro')
    plt.xlabel('Squared Deviation')
    plt.ylabel('Cross Entropy')
    plt.show()
