import math
import numpy as np

"""
    This file contains the Distribution (super)class and its subclasses
    for learning various specific distributions (Gaussian).  
    The Gaussian subclass provides the functionality for training and
    calculating the probability under the model.
    
    written by Eric Nalisnick, enalisnick@gmail.com, Nov 2014
"""


class Distribution(object):
    """ Superclass containing basic distribution functionality """

class Gaussian(Distribution):
    """ Multi-Variate Gaussian (Normal) Distribution """
    def __init__(self, dimension):
        if dimension < 2:
            raise NameError("Error: Gaussian must be at least two dimensional.")
        self.dimension = dimension
        self.__name__ = "%d-D_Gaussian_Distribution" % (self.dimension)
        self.mu = np.zeros((1,self.dimension))
        self.sigma = np.zeros((self.dimension, self.dimension))

    def batch_fit(self, data, weights = None, reset_flag = False):
        """ fit Gaussian via Max Likelihood """
        n,d = data.shape
        self.check_dim(d)
        if weights==None:
            weights = np.ones(n)
        if reset_flag:
            self.mu = np.zeros((1,self.dimension))
            self.sigma = np.zeros((self.dimension, self.dimension))
        for data_idx in xrange(n):
            self.mu += weights[data_idx] * data[data_idx,:]
        self.mu = (1./weights.sum()) * self.mu
        for data_idx in xrange(n):
            self.sigma += weights[data_idx] * np.dot(np.transpose(data[data_idx] - self.mu), data[data_idx] - self.mu)
        self.sigma = (1./weights.sum()) * self.sigma

    def pdf(self, data, log_flag=False):
        d = data.shape[0]
        self.check_dim(d)
        det = np.linalg.det(self.sigma)
        if det == 0:
            raise NameError("Error: The covariance matrix can't be singular.")
        Z = 1.0/ ( math.pow((2*math.pi),float(self.dimension)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(data - self.mu)
        inv = np.matrix(self.sigma).I
        prob_unnorm = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        if log_flag:
            return np.log(Z * prob_unnorm)
        else:
            return Z * prob_unnorm

    def check_dim(self, dimension):
        assert dimension == self.dimension, "Error: Data must be %d-Dimensional." % (self.dimension)