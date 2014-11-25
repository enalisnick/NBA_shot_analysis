from Distributions import *

""" Did this on the fly.  To do: write good ones """

g = Gaussian(2)
data = np.random.rand(100,2)
#g.fit(data)
#print g.pdf(np.array([[4, 4]]))
g.mu = np.array([1,2])
g.sigma = np.array([[1,0],[0,1]])
print g.pdf(np.array([[1, 2]]))
