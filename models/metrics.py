import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

"""
    This file contains various distances metrics.
    
    written by Eric Nalisnick, enalisnick@gmail.com, Nov 2014
"""


class Metric(object):
    """ Superclass containing basic metric functionality """
    def __init__(self, metric):
        self.metric = metric
        self.sim_mat = None

    def calculate_distance_matrix(self, data):
        self.sim_mat = []
        for p0 in data:
            temp_array = []
            for p1 in data:
                temp_array.append(self.metric(p0,p1))
            self.sim_mat.append(temp_array)
        self.sim_mat = np.array(self.sim_mat)

    def visualize(self, labels, output_dir='./'):
        # use Multidimensional Scaling to get 2-D embedding
        seed = np.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=2, metric=True, max_iter=3000,
                           eps=1e-9, random_state=seed,dissimilarity="precomputed", n_jobs=1)
        embedding = mds.fit(self.sim_mat).embedding_
        plt.figure()
        colors = plt.cm.jet(np.linspace(0, 1, self.sim_mat.shape[0]))
        plt.scatter(embedding[:, 0], embedding[:, 1],s=40, c=colors)
        # add player tags
        for label, x, y in zip(labels, embedding[:, 0], embedding[:, 1]):
            plt.annotate(
                        label, xy = (x, y), xytext = (-10, 10),
                         textcoords = 'offset points', ha = 'right', va = 'bottom',
                         bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.5),
                         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.savefig(output_dir+self.__name__+".png", bbox_inches='tight')
        plt.show()

class Gauss_Fisher(Metric):
    """ Fisher information on the 1D Gaussian Manifold """
    def __init__(self):
        self.__name__ = "1D_Gaussian_Fisher_Distance"
        super(Gauss_Fisher, self).__init__(metric = self.manifold_geodesic)

    def manifold_geodesic(self, p0, p1):
        aa = p0[0]-p1[0]
        ab = p0[1]+p1[1]
        bb = p0[1]-p1[1]
        num = np.sqrt((aa**2+ab**2))+np.sqrt((aa**2+bb**2))
        den = np.sqrt((aa**2+ab**2))-np.sqrt((aa**2+bb**2))
        return np.log(num/den)