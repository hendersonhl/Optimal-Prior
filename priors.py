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
    lmda: Lagrange multipliers (T-by-1)
    
"""

import numpy as np
from itertools import permutations, product
from scipy.misc import logsumexp
from scipy.misc import factorial
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import time


class Prior(object):
    def __init__(self, T, N, parms, a, b, corr, proc, sd, z, x0, path):
        """Initializes Prior class."""
        K = len(parms) # number of covariates (including intercept)
        M = len(z) # number of parameter support points
        J = len(z) # number of error support points
        X = self.xdata(T, K, a, b, corr)
        z = np.array(z)[:, np.newaxis]
        u = (1.0/J)*np.ones(T*J)[:, np.newaxis] # uniform prior on noise
        priors = self.priors(M, proc)     
        # run experiment
        self.experiment(N, M, X, u, z, priors, parms, sd, x0, corr, proc, path)    
        # get results
        self.results(path, T, sd, corr, parms, proc)
                
    def experiment(self, N, M, X, u, z, priors, parms, sd, x0, corr, proc, path):
        """Returns experiment results.
        
        Parameters
        ----------
        N : int
            Number of replications
        M : int
            Number of error support points
        X : ndarray
            Covariate matrix
        u : ndarray
            Prior on noise
        z : ndarray
            Parameter support vector
        priors : ndarray
            Possible priors
        parms : ndarray
            Model parameters
        sd : int
            Standard deviation on noise to generate dependent variable
        x0 : ndarray
            Starting values
        corr : int
            Correlated covariates
        proc : int
            Number of coefficients receiving prior procedure 
        path : str
            File path
        
        """
        assert type(N) == type(M) == int, 'N and M must be integers'
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        assert type(sd) == int, 'sd must be integer'
        assert corr==0 or corr==1 or corr==2, 'corr misspecified' 
        assert proc==1 or proc==2, 'proc is misspecified'
        T, K = np.shape(X)
        for i in range(N):
            print 'Enter replication {}'.format(i + 1)
            t0 = time.time()
            y = self.ydata(X, parms, sd)
            v = self.esupport(y, M)
            # define objective function as maximization of dual
            obj = lambda lmda, q: - self.dual(lmda, q, u, y, X, z, v) 
            # define Jacobian
            jac = lambda lmda, q: self.jacobian(lmda, q, u, y, X, z, v)
            coeff, ols, dev_ent, dev_ols, ent = self.fit_model(obj, jac, priors, 
                x0, y, X, u, z, v, parms, proc)     
            data = np.hstack((np.vstack((priors)), np.vstack((coeff)), 
                np.vstack((ols)), np.vstack((dev_ent)), np.vstack((dev_ols)),
                np.vstack((ent))))  
            self.out(path, T, K, M, sd, corr, data, i, proc) # write results 
            print 'Exit replication {0} ({1} seconds wall time)'.format(i + 1,
                time.time() - t0 )
        
    def fit_model(self, obj, jac, priors, x0, y, X, u, z, v, parms, proc):
        """Minimizes dual objective function. 
    
        Parameters
        ----------
        obj : callable func
            Objective function
        jac : callable func
            Jacobian of objective function
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
        proc : int
            Number of coefficients receiving prior procedure     
    
        """ 
        assert np.shape(y)[1] == 1, 'y dimension issue'  
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        assert np.shape(v)[1] == 1, 'v dimension issue'
        assert proc==1 or proc==2, 'proc is misspecified'
        T = len(y)
        K = np.shape(X)[1]
        M = len(z)
        q = (1.0/M)*np.ones(K*M)[:, np.newaxis]
        coeffs, olss, dev_ents, dev_olss, ents = [], [], [], [], []
        for prior in priors:
            toprint = [round(e, 2) for e in prior]
            print 'Estimating model with prior: {}'.format(toprint)
            # introduce iteration-specific prior into q
            q[M: (proc + 1)*M] = np.array(prior)[:, np.newaxis]
            assert np.shape(q)[1] == 1, 'q dimension issue' 
            # fit model
            result = minimize(obj, x0, args=(q,), method='L-BFGS-B', jac=jac, 
                tol=1e-14)
            assert np.isfinite(result.x).all() == True, 'result.x not finite'
            print result.message
            # get output
            pprob = self.pprobs(result.x, y, X, q, z)
            wprob = self.wprobs(result.x, u, v, T)       
            coeff = self.coeffs(pprob, z)
            ols = self.ols(y, X)
            dev_ent = self.sq_dev(parms, coeff.T)
            dev_ols = self.sq_dev(parms, ols.T)
            ent = self.ce(pprob, wprob, q, u)           
            coeffs.append(coeff)
            olss.append(ols)           
            dev_ents.append(dev_ent)
            dev_olss.append(dev_ols)
            ents.append(ent)
        return coeffs, olss, dev_ents, dev_olss, ents
        
    def ols(self, y, X):
        """Returns OLS estimates.
        
        Parameters
        ----------
        y : ndarray
            Dependent variable
        X : ndarray
            Covariates
            
        """
        assert np.shape(y)[1] == 1, 'y dimension issue' 
        p1 = np.linalg.inv(np.dot(X.T, X))
        p2 = np.dot(X.T, y)
        ols = np.dot(p1, p2)       
        assert np.isfinite(ols).all() == True, 'ols not finite'
        return ols.T

    def dual(self, lmda, q, u, y, X, z, v):
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
        assert np.shape(q)[1] == 1, 'q dimension issue'
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(y)[1] == 1, 'y dimension issue'
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
        assert np.isfinite(p2).all() == True, 'p2 not finite'
        p3a = np.dot(V.T, lmda) # TJ-by-1   
        p3b = np.reshape(p3a, (-1, J))
        p3c = logsumexp(p3b, axis=1, b=np.reshape(u, (-1, J)))
        p3 = np.sum(p3c)
        assert type(p3) == np.float64, 'p3 must be float'
        assert np.isfinite(p3).all() == True, 'p3 not finite'
        return p1 - p2 - p3
        
    def jacobian(self, lmda, q, u, y, X, z, v):   
        """Returns Jacobian of dual formulation.

        Parameters
        ----------
        lmda: array
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
        assert np.shape(q)[1] == 1, 'q dimension issue'
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(y)[1] == 1, 'y dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        assert np.shape(v)[1] == 1, 'v dimension issue'
        T = np.shape(X)[0]
        K = np.shape(X)[1]      
        eye_t = np.eye(T)
        eye_k = np.eye(K)
        Z = np.kron(eye_k, z.T)
        V = np.kron(eye_t, v.T)
        pprob = self.pprobs(lmda, y, X, q, z)
        wprob = self.wprobs(lmda, u, v, T)
        return np.squeeze(y - np.dot(np.dot(X, Z), pprob) - np.dot(V, wprob))
              
    def ce(self, pprob, wprob, q, u):
        """Returns cross entropy.
    
        Parameters
        ----------
        pprob : ndarray
            Probabilities on signal
        wprob : ndarray
            Probabilities on noise
        q : ndarray
            Prior on signal 
        u : ndarray
            Prior on noise
    
        """
        assert np.shape(pprob)[1] == 1, 'pprob dimension issue' 
        assert np.shape(wprob)[1] == 1, 'wprob dimension issue'      
        assert np.shape(q)[1] == 1, 'q dimension issue' 
        assert np.shape(u)[1] == 1, 'u dimension issue'          
        assert len(pprob) == len(q), "len(pprob) not equal to len(q)"  
        assert len(wprob) == len(u), "len(wprob) not equal to len(u)"     
        ce_signal = np.sum(pprob * np.log((pprob + 1e-10)/q))
        ce_noise = np.sum(wprob * np.log((wprob + 1e-10)/u))
        ce_total = ce_signal + ce_noise
        assert type(ce_signal) == np.float64, 'ce_signal must be float'
        assert type(ce_noise) == np.float64, 'ce_noise must be float'
        assert type(ce_total) == np.float64, 'ce_total must be float'
        return np.array([[ce_signal, ce_noise, ce_total]])

    def pprobs(self, lmda, y, X, q, z):
        """Returns probabilities on signal.
    
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
        p1b = logsumexp(p1a, axis=1, b=q)
        p1 = np.exp(p1b)[:,np.newaxis]
        #p1b = np.exp(p1a)
        #p1 = q * p1b
        assert np.isfinite(p1).all() == True, 'p1 not finite'
        assert np.shape(p1) == (K*M, 1), 'p1 dimension issue'
        p2a = np.dot(ones_m, ones_m.T)
        p2b = np.kron(eye_k, p2a)
        p2 = np.dot(p2b, p1)
        assert np.shape(p2) == (K*M, 1), 'p2 dimension issue'
        assert np.isfinite(p2).all() == True, 'p2 not finite'
        return p1 / p2  
    
    def wprobs(self, lmda, u, v, T):
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
        p1b = logsumexp(p1a, axis=1, b=u)
        p1 = np.exp(p1b)[:,np.newaxis]        
        #p1b = np.exp(p1a)
        #p1 = u * p1b
        assert np.shape(p1) == (T*J, 1), 'p1 dimension issue'
        assert np.isfinite(p1).all() == True, 'p1 not finite'
        p2a = np.dot(ones_j, ones_j.T)
        p2b = np.kron(eye_t, p2a)
        p2 = np.dot(p2b, p1)
        assert np.shape(p2) == (T*J, 1), 'p2 dimension issue'
        assert np.isfinite(p2).all() == True, 'p2 not finite'
        return p1 / p2 
            
    def xdata(self, T, K, a, b, corr):
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
        corr : int
            Correlated covariates
    
        """
        assert T > 0 and K > 0, 'inputs must be non-negative'
        assert a >= 0 and b > 0, 'inputs must be non-negative'
        assert b > a, 'a must be less than b'
        assert corr==0 or corr==1 or corr==2, 'corr misspecified'       
        if corr==0: # uncorrelated covariates
            X1 = np.random.uniform(a, b, size=(T, K - 1)) 
            X = np.hstack((np.ones(T)[:, np.newaxis], X1))
        elif corr==1:  # one pair correlated
            assert K > 2, 'if corr==1, K must be greater than two'
            X1 = np.random.uniform(a, b, size=(T, K - 2)) 
            X2 = 2*X1[:,0][:,np.newaxis] + np.random.normal(0, 5, size=(T, 1))
            X = np.hstack((np.ones(T)[:, np.newaxis], X1, X2))        
        else: # two pairs correlated
            assert K > 5, 'if corr==2, K must be greater than five'
            X1 = np.random.uniform(a, b, size=(T, K - 3)) 
            X2 = 2*X1[:,0][:,np.newaxis] + np.random.normal(0, 5, size=(T, 1))            
            X3 = -3*X1[:,2][:,np.newaxis] + np.random.normal(0, 5, size=(T, 1))
            X = np.hstack((np.ones(T)[:, np.newaxis], X1, X2, X3))  
        assert np.shape(X) == (T, K), 'X dimension issue'
        return X
    
    def ydata(self, X, parms, sd):
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

    def priors(self, M, proc):
        """Returns all possible priors.
    
        Parameters
        ----------
        M : int
            Number of support points
        proc : int
            Number of coefficients receiving prior procedure
        
        """
        assert M > 0, 'M must be greater than zero'
        assert type(M) == int, 'M must be an integer'
        assert proc==1 or proc==2, 'proc is misspecified'
        seq = xrange(1, M + 1)
        prior = []
        for perm in permutations(seq, M):
            normalization = (M * (M + 1)) / 2.
            prior.append(np.asarray(perm)/normalization)
        # uniform prior last
        prior.append(np.ones(M) / M)
        if proc==2: # conduct prior procedure on two coefficients
            prod = product(prior, prior)
            prior = [np.concatenate(i) for i in prod]
        return prior

    def esupport(self, y, M):
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
        
    def coeffs(self, probs, z):
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
        return np.dot(Z, probs).T
    
    def sq_dev(self, parms, coeffs):
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
        sq_dev = (parms - coeffs)**2
        sq_dev = np.vstack((sq_dev, sq_dev.sum())) # add total squared dev.
        assert np.shape(sq_dev)[1] == 1, 'sq_dev dimension issue'
        return sq_dev.T 
 
    def out(self, path, T, K, M, sd, corr, data, n, proc):
        """Writes model output to file. 
        
        Parameters
        ----------
        path : str
            File path
        T : int
            Number of observations
        K : int
            Number of covariates (including intercept)
        M : int
            Number of error support points
        sd : int
            Standard deviation on noise to generate dependent variable
        corr : int
            Correlated covariates
        data : ndarray
            Data to be written to file
        n : int
            Replication number
        proc : int
            Number of coefficients receiving prior procedure 
            
        """
        assert type(T) == type(K) == int, 'T and K must be integers'
        assert type(M) == type(sd) == int, 'M and sd must be integers'
        assert corr==0 or corr==1 or corr==2, 'corr misspecified' 
        assert proc==1 or proc==2, 'proc is misspecified'
        nstr = str(T) + str(K) + str(sd) + str(corr) + str(proc)
        fname = path + 'out' + nstr + '.csv'
        prior, b, ols, dev_ent, dev_ols = [], [], [], [], []      
        if n==0: # write header and data
            with open(fname, 'w') as f:
                # write header
                for i in range(proc*M):
                    prior.append('prior' + str(i))
                for i in range(K):
                    b.append('b' + str(i))
                    ols.append('ols' + str(i))
                    dev_ent.append('dev_ent' + str(i))
                    dev_ols.append('dev_ols' + str(i))
                dev_ent.append('dev_ent')
                dev_ols.append('dev_ols')
                ce = ['ce_signal', 'ce_noise', 'ce_total']    
                header = prior + b + ols + dev_ent + dev_ols + ce
                f.write(','.join(header))
                # write data
                for i in data:
                    strdata = [str(j) for j in i]
                    f.write('\n' + ','.join(strdata))
        else: # just write data
            with open(fname, 'a') as f:
                for i in data:
                    strdata = [str(j) for j in i]
                    f.write('\n' + ','.join(strdata))
                    
    def results(self, path, T, sd, corr, parms, proc):
        """Creates post-simulation tables and graphs.
        
        Parameters
        ----------
        path : str
            File path
        T : int
            Number of observations
        sd : int
            Standard deviation on noise to generate dependent variable
        corr : int
            Correlated covariates
        parms : ndarray
            Model parameters
        proc : int
            Number of coefficients receiving prior procedure 
        
        """
        assert type(T) == type(sd) == int, 'T and sd must be integers' 
        assert proc==1 or proc==2, 'proc is misspecified'
        K = len(parms)        
        nstr = str(T) + str(K) + str(sd) + str(corr) + str(proc)
        data = path + 'out' + nstr + '.csv'
        mresults = path + 'mresults' + nstr + '.csv'
        vresults = path + 'vresults' + nstr + '.csv'
        figure = path + 'figure' + nstr
        df = pd.read_csv(data) 
        # group by prior
        prior = [i for i in df.columns if i.startswith('prior')]
        grouped = df.groupby(prior) 
        means = grouped.agg(np.mean) # aggregate by means
        variances = grouped.agg(np.var) # aggregate by variances
        # bias calculations 
        for i in range(K): # for ce coefficients
            bias_ent = 'bias_ent' + str(i)
            b = 'b' + str(i)
            means[bias_ent] = means[b] - parms[i]
        for i in range(K): # for ols coefficients
            bias_ols = 'bias_ols' + str(i)
            ols = 'ols' + str(i)
            means[bias_ols] = means[ols] - parms[i]
        means.to_csv(mresults) # write to csv
        variances.to_csv(vresults) # write to csv          
        # plot results
        if len(prior)==3:
            marker = ['b', 'g', 'r', 'c', 'y', 'k', 'w'] 
        else:
            npriors = int(factorial(len(prior)) + 1)
            marker = [tuple(np.random.uniform(0, 1, size=(1, 3))[0]) 
                for i in range(npriors)] # list of RGB tuples  
        lst = [('dev_ent','ce_total', 'all'),('dev_ent','ce_total','means'),
            ('dev_ent1','ce_total','all'),('dev_ent1','ce_total','means'),
            ('dev_ent','ce_signal', 'all'),('dev_ent','ce_signal','means'),
            ('dev_ent1','ce_signal','all'),('dev_ent1','ce_signal','means')]
        for i in lst:
            plt.figure()
            counter = 0
            for key, grp in grouped:                       
                temp1 = [round(j, 2) for j in key] # two decimal places               
                temp2 = str(tuple(temp1))
                if i[2]=='all':
                    plt.scatter(grp[i[0]], grp[i[1]], marker='o', 
                        c=marker[counter], label=temp2)
                else:
                    plt.scatter(np.mean(grp[i[0]]), np.mean(grp[i[1]]), 
                        marker='o', c=marker[counter], label=temp2)
                counter +=1
            plt.xlabel('Squared Deviation')
            plt.ylabel('Cross Entropy')
            if proc==1:
                lgd = plt.legend(scatterpoints=1, bbox_to_anchor=(1.35,1), 
                    fontsize='medium') 
            else:
                lgd = plt.legend(scatterpoints=1, bbox_to_anchor=(1.81,1), 
                    fontsize='x-small', ncol=2) 
            plt.savefig(figure + '(' + i[0] + ')' + '(' + i[1] + ')' +
                '(' + i[2] + ')' + '.png', bbox_extra_artists=(lgd,), 
                bbox_inches='tight')                        
                                 
if __name__ == "__main__":

    # set seed
    np.random.seed(12345)

    # user inputs
    T = 10 # sample size: [10, 20, 50, 100, 250, 500]
    N = 2 # number of replications: [100, 1000, 5000]
    parms = [1.0, -5.0, 2.0] # parameter values
    #parms = [1.0, -5.0, 2.0, -3.0, 8.0, 6.0, -2.0, -7.0, 4.0, -1.0] 
    a = 0 # lower bound on uniform dist. of covariates
    b = 20 # upper bound on uniform dist. of covariates
    corr = 0 # pairs of correlated covariates: [0, 1, 2]
    proc = 1 # number of coefficients receiving prior procedure: [1, 2]
    sd = 2 # standard deviation on model noise
    z = [-200.0, 0, 200.0] # support for parameters
    x0 = np.zeros(T) # starting values
    path = '/Users/hendersonhl/Documents/Articles/Optimal-Prior/Output/'
    
    # run experiment
    exp = Prior(T, N, parms, a, b, corr, proc, sd, z, x0, path)

