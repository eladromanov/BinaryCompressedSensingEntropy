# -*- coding: utf-8 -*-

import numpy as np
from math import floor
from numpy.random import permutation, choice, rand, randint



"""
Generate the parity check matrix of a random Gallager code.
Parameters:
    t: variable degree
    s: factor degree
    n: length of codeword (number of variables)
    seed:  
Returns:
    variables: a list of lists, such that variables[i] are all the parities connected to variable i
    factors: a list of lists, such that factors[j] are all the variables connected to factor j
    
"""
def makeGallagerMatrix(t, s, n, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    factors = []
    for it in range(0,t):
        perm = permutation(n)
        j = 0 
        while j*s < n:
            factors.append( list(perm[j*s:j*s+s]) )
            j = j + 1
    
    variables = []
    for i in range(0,n):
        variables.append(list())
    for j in range(0,len(factors)):
        for i in factors[j]:
            variables[i].append(j)
            
            
    np.random.seed()
            
    return variables, factors


"""
Returns y=Ax, where x is a vector and A is a sparse matrix defined by a list of factors - factors is a python list of lists.
factors[j] are all the variables indices that are connected to factor j.
Thus,
y[j] = \sum_{i in factors[j]} x[i]

Returns:
    y: a numpy array
"""
def multiplyBySparseMatrix(factors, x):
    y = np.zeros( len(factors) )
    for j in range(0, len(factors) ):
        y[j] = np.sum( x[factors[j]] )
    return y


"""
Recover a binary 0-1, n-dimensional vector x from measurements y=Ax+sigma*z, by Gibbs sampling
(Glauber dynamics) from the posterior. 
Parameters:
    y - an m-dimensional vector (numpy array) of measurements
    variables, factors - lists of lists describing a binary sensing matrix
    sigma - noise standard deviation to use
    prior - a vector with a product prior distribution for x: prior[i] = Pr(x_i=1)
    annealing (optional) - a multiplicative factor by which to exponentiate 
        the Gibbs distribution. A number < 1.0 potentially gives faster mixing time, at the 
        expense of statistical accuracy. default=1.0
    T (int, optional) - number of iterations to use. default=5*n*log(n)
    initX (optional) - numpy array; an initial (assumed binary) vector to start the chain form. If none is given, starts from the all zeros state
Returns:
    X_T - the last state attained by the chain
    softEst - an n-dimensional numpy array of soft decisions. softEst is the conditional distribution that 
        x[i]=1, conditioned on all the other coordinates being X_T, where X_T is the vector to which 
        we converged.

"""
def glauberDynamics_Unoptimized(y, variables, factors, sigma, prior, annealing=1.0, T=None, initX=None):
    n = len(variables)
    if T is None:
        T = int( 5*n*np.log(n) )   
    
    X_t = np.zeros(n)
    if initX is not None:
        X_t = initX
    
    # pre-generate all randomness, to hopefully make this a bit faster
    random_indices = randint(0, n, size=T)
    random_coins = rand(T)
    
    # run the chain
    
    for t in range(0,T):
        
        indx = random_indices[t]
        X_t[indx] = 0
        
        relevant_parities = variables[indx]    # parities connect to variable indx, and hence participate in conditional probability
        
        val0 = val1 = 0.0   # val0 (similarly val1) is, up to normalization, log the probability that X[indx]=0 conditioned on its neighbors
        
        for f in relevant_parities:
            sum_f = np.sum( X_t[ factors[f] ] )
            val0 = val0 - (y[f]-sum_f)**2/(2*sigma**2)
            val1 = val1 - (y[f]-sum_f-1)**2/(2*sigma**2)
        val1 = val1 + np.log(prior[indx]/(1-prior[indx]))
        
        # apply annealing, if necessary
        val0 = annealing*val0
        val1 = annealing*val1
        
        p1 = 1/(np.exp(val0-val1)+1)
        if random_coins[t] <= p1:
            X_t[indx] = 1
       
    
    # after letting the chain mix, produce a "soft" rule. softEst[i] is the conditional 
    # probability that x_i=1, given that all its neighbors are as X_t
    
    softEst = np.zeros(n)
    
    for i in range(0,n):
        
        relevant_parities = variables[i]
        val0 = val1 = 0.0
        
        for f in relevant_parities:
            sum_f = np.sum( X_t[ factors[f] ] ) - X_t[i] # we condition on all coordinates but i; that's why we need to substract X_t[i] here
            val0 = val0 - (y[f]-sum_f)**2/(2*sigma**2)
            val1 = val1 - (y[f]-sum_f-1)**2/(2*sigma**2)
        val1 = val1 + np.log(prior[i]/(1-prior[i]))
        
        # apply annealing, if necessary
        val0 = annealing*val0
        val1 = annealing*val1
        
        p1 = 1/(np.exp(val0-val1)+1)
        softEst[i] = p1
        
            
    return X_t, softEst


def glauberDynamics(y, variables, factors, sigma, prior, annealing=1.0, T=None, initX=None):
    
    n = len(variables)
    m = len(factors)
    if T is None:
        T = int( 5*n*np.log(n) )   
    
    X_t = np.zeros(n)
    if initX is not None:
        X_t = initX
    
    Y_t = np.zeros(m)
    for j in range(0,m):
        Y_t[j] = np.sum( X_t[factors[j]] )
    # Keep Y_t = A X_t
    
    # pre-generate all randomness, to hopefully make this a bit faster
    random_indices = randint(0, n, size=T)
    random_coins = rand(T)
    
    # run the chain
    
    for t in range(0,T):
        
        indx = random_indices[t]
        relevant_parities = variables[indx]    # parities connect to variable indx, and hence participate in conditional probability
        
        if X_t[indx] == 1:
            Y_t[ relevant_parities ] = Y_t[ relevant_parities ]-1
        X_t[indx]=0
        
        diff = y[relevant_parities] - Y_t[relevant_parities]
        val0 = -np.linalg.norm(diff)**2/ (2*sigma**2)
        val1 = -np.linalg.norm(diff-1)**2/ (2*sigma**2) + np.log(prior[indx]/(1-prior[indx]))
        
        # apply annealing, if necessary
        val0 = annealing*val0
        val1 = annealing*val1
        
        p1 = 1/(np.exp(val0-val1)+1)
        if random_coins[t] <= p1:
            X_t[indx] = 1
            Y_t[ relevant_parities ] = Y_t[ relevant_parities ]+1
       
    
    # after letting the chain mix, produce a "soft" rule. softEst[i] is the conditional 
    # probability that x_i=1, given that all its neighbors are as X_t
    
    softEst = np.zeros(n)
    
    for i in range(0,n):
        
        relevant_parities = variables[i]
        
        diff = y[relevant_parities] - Y_t[relevant_parities] + X_t[i]
        val0 = -np.linalg.norm(diff)**2/ (2*sigma**2)
        val1 = -np.linalg.norm(diff-1)**2/ (2*sigma**2) + np.log(prior[i]/(1-prior[i]))
        
        # apply annealing, if necessary
        val0 = annealing*val0
        val1 = annealing*val1
        p1 = 1/(np.exp(val0-val1)+1)
        softEst[i] = p1
        
            
    return X_t, softEst


"""
Matching pursuit, to cheaply get an initial guess for Glauber
"""
def matchingPursuit(y, variables, factors, K):
    
    n = len(variables)
    corr = np.zeros(n)
    for i in range(0,n):
        corr[i] = np.sum( y[ variables[i] ]  )
        
    active = []
    for t in range(0,np.min([K,n])):
        
        i_new = np.argmax(corr)     # actually don't care if there are repeats
        active.append(i_new)
        
        # Now need to update the correlations; we remove the i-th codeword X_i from 
        # the measurement y: y <- y - X_i .
        # Thus, we need to update the correlations as corr[j] <- corr[j] - <X_i,X_j>.
        # Note that the only affected variables j are j=i or neighbors of i (meaning, they share a factor)
        
        vars_to_update = [i_new]
        for f in variables[i_new]:
            vars_to_update = np.union1d(vars_to_update, factors[f])
        for j in vars_to_update:
            corr[j] = corr[j] - np.intersect1d(variables[i_new],variables[j]).size
            
    X_est = np.zeros(n)
    X_est[active] = 1.0
    return X_est


"""
Glauber dynamics with a dense sensing matrix.
"""
def Glauber_Dense(y, A, sigma, prior, T=None, initX=None):
    
    _, n = A.shape
    if T is None:
        T = 2*n*np.log(n)
    
    X_t = initX
    if initX is None:
        X_t = np.zeros(n)
    Y_t = A @ X_t
    
    random_indices = randint(0, n, size=T)
    random_coins = rand(T)
    
    for t in range(0,T):
        
        indx = random_indices[t]
        
        Ywith0 = Y_t
        Ywith1 = Y_t
        if X_t[indx] == 0:
            Ywith1 = Y_t + A[:,indx]
        else:
            Ywith0 = Y_t - A[:,indx]
                
        val0 = -np.linalg.norm(y-Ywith0)**2/ (2*sigma**2)
        val1 = -np.linalg.norm(y-Ywith1)**2/ (2*sigma**2) + np.log(prior[indx]/(1-prior[indx]))
        p1 = 1/(np.exp(val0-val1)+1)
        
        if random_coins[t] <= p1: # Set coordinate indx to 1
            if X_t[indx] !=  1:
                X_t[indx] = 1   
                Y_t = Ywith1
            
        else:   # Set coordinate indx to 0
            if X_t[indx] != 0:
                X_t[indx] = 0
                Y_t = Ywith0
        
    return X_t
    
    