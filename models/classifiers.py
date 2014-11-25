import datetime as dt
from distributions import *
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import sklearn.neighbors

"""
    This file contains the Classifer (super)class and its subclasses
    for training various specific classifiers (k-Nearest Neighbors,
    Logistic Regression, Gaussian Classifier, Mixture of Gaussians,
    Support Vector Machine).  The base class provides the functionality
    for training, testing, and visualizing the decision boundaries of a
    classifier.  
    
    The code is generally applicable except for the visualization functionality,
    which assumes we wish to see the class boundaries superimposed on
    a regulation NBA half-court.  Some variables have names such as 'missed_
    shot_data' but are not fundamentally restricted to that type of data.
    
    written by Eric Nalisnick, enalisnick@gmail.com, Nov 2014
"""

class Classifier(object):
    """ Superclass containing basic classifier functionality """
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, training_data, training_labels):
        self.classifier.fit(training_data, training_labels)
    
    def test(self, test_data, test_labels):
        """ 
            Tests the classifier on a held-out set
            and returns a tuple containing two values:
            the accuracy on the test set and the intrinsic
            lowerbound on the accuracy (ie 50% in a balanced
            two class problem)
        """
        num_correct = 0.
        baseline_num_missed = 0.
        n = test_data.shape[0]
        predictions = self.use(test_data)
        for idx in xrange(n):
            if predictions[idx] == test_labels[idx]:
                num_correct += 1
            if test_labels[idx] == 0:
                baseline_num_missed += 1
        accuracy = num_correct/n
        miss_baseline_accuracy = baseline_num_missed/n
        # log results
        self.write_to_log("Test Accuracy: %f\n" %(accuracy))
        self.write_to_log("Miss Baseline Accuracy: %f\n\n" %(miss_baseline_accuracy))
        return (accuracy, miss_baseline_accuracy)

    def use(self, test_data):
        return self.classifier.predict(test_data)

    def visualize(self, court_image = './data/nba_court.jpg', output_dir = './'):
        """
            This function visualizes the classifier's make/miss decision boundary
            on a regulation NBA half-court--the dimensions are hard-coded.  A lattice
            is generated and every point in it classified.  This isn't optimal for 
            some classifiers (the ones with analytical solutions) but isn't a prolem
            here since the space of points residing on the half-court is relatively small.
            A court image and output directory can be specified.
        """
        two_class_cmap = ListedColormap(['#FFAAAA', '#AAFFAA']) # light red for miss, light green for make
        x_min, x_max = 0, 50 #width (feet) of NBA court
        y_min, y_max = 0, 47 #length (feet) of NBA half-court
        grid_step_size = 0.2
        grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, grid_step_size), np.arange(y_min, y_max, grid_step_size))
        grid_predictions = self.use(np.c_[grid_x.ravel(), grid_y.ravel()])
        grid_predictions = grid_predictions.reshape(grid_x.shape)
        fig, ax = plt.subplots()
        court_image = plt.imread(court_image)
        ax.imshow(court_image, interpolation='bilinear', origin='lower',extent=[x_min,x_max,y_min,y_max])
        ax.imshow(grid_predictions, cmap=two_class_cmap, interpolation = 'nearest',
                  alpha = 0.60, origin='lower',extent=[x_min,x_max,y_min,y_max])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        #ax.axes.get_xaxis().set_visible(False)
        #ax.axes.get_yaxis().set_visible(False)
        #plt.title( "Make / Miss Boundaries for %s" % (self) )
        plt.savefig(output_dir+self.__name__+".png", bbox_inches='tight')

    def write_to_log(self, message, log_file_name=None):
        """ Function for logging training progress and test results """
        if log_file_name==None:
            log_file_name = self.__name__+'.log.txt'
        with open(log_file_name, 'a') as log_file:
            log_file.write(message)

#################################################################

class kNN(Classifier):
    """ 
        k-Nearest Neighbors Subclass (wrapper for scikit-learn)
        Parameters: k, the numer of neighbors to examine
        See Scikit-Learn documentation
    """
    def __init__(self, k):
        self.k = k
        self.__name__ = "k-Nearest_Neighbors"
        super(kNN, self).__init__( classifier =
                                  sklearn.neighbors.KNeighborsClassifier( self.k, weights='distance' ) )
        self.write_to_log("%s initialized at %s\n\n"%(self,dt.datetime.now()))

    def __str__(self):
        return "k-Nearest Neighbors Classifier (k=%d)" %(self.k)

#################################################################

class LogisticRegression(Classifier):
    """ 
        Logistic Regression Subclass (wrapper for scikit-learn)
        See Scikit-Learn documentation
    """
    def __init__(self):
        self.__name__ = "Logistic_Regression"
        super(LogisticRegression, self).__init__( classifier =
                                  sklearn.linear_model.LogisticRegression( C=1e5 ) )
        self.write_to_log("%s initialized at %s\n\n"%(self,dt.datetime.now()))
    
    def __str__(self):
        return "Logistic Regression Classifier"

#################################################################

class SupportVectorMachine(Classifier):
    """
        Support Vector Machine (SVM) Subclass (wrapper for scikit-learn)
        See Scikit-Learn documentation
        """
    def __init__(self):
        self.__name__ = "Support_Vector_Machine"
        super(SupportVectorMachine, self).__init__( classifier =
                                                 sklearn.svm.SVC( kernel="poly" ) )
        self.write_to_log("%s initialized at %s\n\n"%(self,dt.datetime.now()))
    
    def __str__(self):
        return "Support Vector Machine Classifier"

#################################################################

class Gaussian2DClassifier(Classifier):
    """ 
        2D Gaussian Classifier Subclass
    """
    def __init__(self):
        self.__name__ = "2D_Gaussian_Classifier"
        self.made_gauss = Gaussian(2) #fit one Gaussian to made shots
        self.missed_gauss = Gaussian(2) #fit another to missed shots
        self.write_to_log("%s initialized at %s\n\n"%(self,dt.datetime.now()))

    def train(self, made_shot_training_data, missed_shot_training_data):
        """
            Training functionality.  Given two datasets, one containing
            missed shots and another made shots, fit a Gaussian to each
        """
        self.made_gauss.batch_fit(made_shot_training_data)
        self.missed_gauss.batch_fit(missed_shot_training_data)
        self.write_to_log("Training complete.\n")
        self.write_to_log("Made Gauss Params: mu=[%.2f,%.2f], cov=[%.2f,%.2f;%.2f,%.2f]\n"
                          %(self.made_gauss.mu[0,0],self.made_gauss.mu[0,1],
                            self.made_gauss.sigma[0][0],self.made_gauss.sigma[0][1],
                            self.made_gauss.sigma[1][0],self.made_gauss.sigma[1][1]))
        self.write_to_log("Missed Gauss Params: mu=[%.2f,%.2f], cov=[%.2f,%.2f;%.2f,%.2f]\n\n"
                          %(self.missed_gauss.mu[0,0],self.missed_gauss.mu[0,1],
                            self.missed_gauss.sigma[0][0],self.missed_gauss.sigma[0][1],
                            self.missed_gauss.sigma[1][0],self.missed_gauss.sigma[1][1]))

    def use(self, test_data):
        """
            Inference functionality.  Given a datapoint, predict the class whose Gaussian
            assigns the point a higher probability.
        """
        n,d = test_data.shape
        predictions = np.zeros(n)
        for idx in xrange(n):
            if self.made_gauss.pdf(test_data[idx,:]) >= self.missed_gauss.pdf(test_data[idx,:]):
                predictions[idx] = 1
        return predictions

    def __str__(self):
        # printing the covariance matrix makes this string too long
        return "2-D Gaussian Classifier (made:mu=[%0.1f,%0.1f] / missed:mu=[%0.1f,%0.1f])" %(self.made_gauss.mu[0,0],self.made_gauss.mu[0,1],self.missed_gauss.mu[0,0],self.missed_gauss.mu[0,1])

#################################################################

class GaussianMixtureClassifier(Classifier):
    """ 
        Gaussian Mixture Model Classifier Subclass 
        Parameters: num_of_mixtures, the number of 
        Gaussians to include in the mixture.
    """
    def __init__(self, num_of_mixtures):
        self.num_of_mixtures = num_of_mixtures
        self.__name__ = "Gaussian_Mixture_Model"
        self.made_mixtures = [] # Gausses for made shots
        self.missed_mixtures = [] # Gausses for missed shots
        for idx in xrange(num_of_mixtures):
            self.made_mixtures.append(Gaussian(2))
            self.missed_mixtures.append(Gaussian(2))
        self.write_to_log("%s initialized at %s\n\n"%(self,dt.datetime.now()))

    def train(self, made_shot_training_data, missed_shot_training_data, max_iterations=20):
        """
            Training functionality.  Given missed and made shot datasets, run EM to fit a
            mixture of Gaussians to each.
            Note: Performance may be an issue for high-dimensional data.
        """
        # We should be re-running EM using several random initializations--to do.
        self.run_EM(self.made_mixtures, made_shot_training_data, max_iterations)
        self.run_EM(self.missed_mixtures, missed_shot_training_data, max_iterations)
        self.write_to_log("Training complete.\n")
        for idx in xrange(self.num_of_mixtures):
            self.write_to_log("Made Gauss #%d Params: mu=[%.2f,%.2f], cov=[%.2f,%.2f;%.2f,%.2f]\n"
                          %(idx, self.made_mixtures[idx].mu[0,0],self.made_mixtures[idx].mu[0,1],
                            self.made_mixtures[idx].sigma[0][0],self.made_mixtures[idx].sigma[0][1],
                            self.made_mixtures[idx].sigma[1][0],self.made_mixtures[idx].sigma[1][1]))
            self.write_to_log("Missed Gauss #%d Params: mu=[%.2f,%.2f], cov=[%.2f,%.2f;%.2f,%.2f]\n"
                          %(idx, self.missed_mixtures[idx].mu[0,0],self.missed_mixtures[idx].mu[0,1],
                          self.missed_mixtures[idx].sigma[0][0],self.missed_mixtures[idx].sigma[0][1],
                          self.missed_mixtures[idx].sigma[1][0],self.missed_mixtures[idx].sigma[1][1]))
    
    def use(self, test_data):
        """
            Inference functionality.  Given a datapoint, predict the class whose Gaussian
            assigns the point the highest probability.
        """
        n,d = test_data.shape
        predictions = np.zeros(n)
        for data_idx in xrange(n):
            made_prob = []
            missed_prob = []
            for mixture_idx in xrange(self.num_of_mixtures):
                made_prob.append(self.made_mixtures[mixture_idx].pdf(test_data[data_idx,:]))
                missed_prob.append(self.missed_mixtures[mixture_idx].pdf(test_data[data_idx,:]))
            if max(made_prob) >= max(missed_prob):
                predictions[data_idx] = 1
        return predictions

    def run_EM(self, gauss_array, training_data, max_iterations=20, epsilon = 0.0000000009):
        """
            The Expectation-Maximization (EM) Algorithm
            See http://melodi.ee.washington.edu/~bilmes/mypubs/bilmes1997-em.pdf
            for details.
        """
        n,d = training_data.shape
        last_logL = 0.
        current_logL = -1000000000.
        iteration_idx = 0
        converged = False
        self.write_to_log("Log-Likelihood Progress:\n")
        # randomly initialize memberships
        memberships = np.random.rand(n, self.num_of_mixtures)
        memberships = memberships / memberships.sum(axis=1)[:, np.newaxis]
        alpha = self.M_step(gauss_array, memberships, training_data)
        while iteration_idx<max_iterations and converged==False:
            # perform E-step
            memberships = self.E_step(gauss_array, alpha, training_data)
            # perform M-step
            alpha = self.M_step(gauss_array, memberships, training_data)
            # calculate log-likelihood under current model
            current_logL = 0.
            for data_point in training_data:
                temp_logL = 0.
                for mixture_idx in xrange(self.num_of_mixtures):
                    temp_logL += alpha[mixture_idx] * gauss_array[mixture_idx].pdf(data_point)
                current_logL += np.log(temp_logL)
            self.write_to_log("Iteration #%d: %.4f\n"%(iteration_idx,current_logL))
            if abs(current_logL-last_logL)<epsilon:
                break
            last_logL = current_logL
            iteration_idx += 1
        return current_logL

    def M_step(self, gauss_array, memberships, data):
        # Maximization Step: Do a weighted Maximum-Likelihood fit
        n,d = data.shape
        N = memberships.sum(axis=0)
        alpha = N / N.sum()
        for mixture_idx in xrange(self.num_of_mixtures):
            gauss_array[mixture_idx].batch_fit(data, memberships[:,mixture_idx], reset_flag=True)
        return alpha

    def E_step(self, gauss_array, alpha, data):
        # Expectation Step: Compute new mixture weights based on new model params
        n,d = data.shape
        memberships = np.zeros((n, self.num_of_mixtures))
        for data_idx in xrange(n):
            for component_idx1 in xrange(self.num_of_mixtures):
                Z = 0.
                for component_idx2 in xrange(self.num_of_mixtures):
                    Z += alpha[component_idx2] * gauss_array[component_idx2].pdf(data[data_idx,:])
                memberships[data_idx, component_idx1] = (1./Z) * alpha[component_idx1] * gauss_array[component_idx1].pdf(data[data_idx,:])
        return memberships

    def __str__(self):
        return "Mixture of 2-D Gaussians Classifier (k=%d)" %(self.num_of_mixtures)


