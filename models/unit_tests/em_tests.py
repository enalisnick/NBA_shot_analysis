import numpy as np
from classifiers import *

data = np.random.rand(100,2)
memberships = np.random.rand(100,3)
print data.shape
gm = GaussianMixtureClassifier(3)
#m_step
n,d = data.shape
N = memberships.sum(axis=0)
print N
alpha = N / N.sum()
print alpha
print alpha.sum()

print memberships[:,1][:,np.newaxis]*data

#for mix_idx in xrange(3):
    
