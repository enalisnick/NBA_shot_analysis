from data.data_accessors import *
import numpy as np
from models.classifiers import *

"""
    This file contains the logic for experiment #1 (prediction of shot
    outcome using only spatial features), experiment #2 (prediction
    based on location and spatial features), and experiment #3 (using
    spatial, position, and shot-type features)--but not the neural
    network component of #3.
    Running this file will output accuracy results to logs created in
    the current directory.  Per-position visualizations will be placed
    in the following directory structure which will assume exists:
    ./position_graphs/[position abbreviation]/

    written by Eric Nalisnick, enalisnick@gmail.com, Nov 2014
"""

def experiment_1(train_seasons, test_seasons):
    """ Experiment #1: Use only spatial features for prediction """
    # load training data
    made_data, missed_data = load_seasons(train_seasons, split_flag=True)
    train_features, train_labels = load_seasons(train_seasons)
    
    # load test data
    test_features, test_labels = load_seasons(test_seasons)
    
    ###### non-parametric models #####
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.train(train_features, train_labels)
    log_reg.test(test_features, test_labels)
    log_reg.visualize()

    # k-Nearest Neighbors
    num_of_neighbors = 5
    kNN_model = kNN(num_of_neighbors)
    kNN_model.train(train_features, train_labels)
    kNN_model.test(test_features, test_labels)
    kNN_model.visualize()

    ###### parametric models #####
    # 2-D Gaussian Classifier
    gauss_model = Gaussian2DClassifier()
    gauss_model.train(made_data, missed_data)
    gauss_model.test(test_features, test_labels)
    gauss_model.visualize()

    # Mixture of 2-D Gaussians Classifier
    num_of_mixtures = 3
    gauss_mixture_model = GaussianMixtureClassifier(num_of_mixtures)
    gauss_mixture_model.train(made_data, missed_data, 10)
    gauss_mixture_model.test(test_features, test_labels)
    gauss_mixture_model.visualize()

def experiment_2(train_seasons, test_seasons):
    """ Experiment #2: Use spatial and position features for prediction """
    
    positions = ['G', 'F', 'C']
    
    for position in positions:
        # load training data
        made_data, missed_data = load_seasons(seasons=train_seasons, split_flag=True, attributes=[position])
        train_features, train_labels = load_seasons(seasons=train_seasons, attributes=[position])
        # load test data
        test_features, test_labels = load_seasons(seasons=test_seasons, attributes=[position])

        output_directory = "./position_graphs/%s/"%(position)
        ###### non-parametric models #####
        # Logistic Regression
        log_reg = LogisticRegression()
        log_reg.train(train_features, train_labels)
        log_reg.test(test_features, test_labels)
        log_reg.visualize(output_dir=output_directory)
    
        # k-Nearest Neighbors
        num_of_neighbors = 5
        kNN_model = kNN(num_of_neighbors)
        kNN_model.train(train_features, train_labels)
        kNN_model.test(test_features, test_labels)
        kNN_model.visualize(output_dir=output_directory)
    
        ###### parametric models #####
        # 2-D Gaussian Classifier
        gauss_model = Gaussian2DClassifier()
        gauss_model.train(made_data, missed_data)
        gauss_model.test(test_features, test_labels)
        gauss_model.visualize(output_dir=output_directory)
    
        # Mixture of 2-D Gaussians Classifier
        num_of_mixtures = 3
        gauss_mixture_model = GaussianMixtureClassifier(num_of_mixtures)
        gauss_mixture_model.train(made_data, missed_data, 10)
        gauss_mixture_model.test(test_features, test_labels)
        gauss_mixture_model.visualize(output_dir=output_directory)

def experiment_3(train_seasons, test_seasons):
    """ Experiment #3: Use spatial, position, and shot type features """
    base_positions = ['G', 'F', 'C']
    base_shot_types = ['3pt','fade away', 'hook', 'layup', 'jump', 'dunk']
    pos_and_shot_type = base_positions + base_shot_types
    
    # load training data
    train_features, train_labels = load_seasons(train_seasons, attributes=pos_and_shot_type, with_attributes_flag=True)
    
    # load test data
    test_features, test_labels = load_seasons(test_seasons, attributes=pos_and_shot_type, with_attributes_flag=True)
    
    ###### non-parametric models #####
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.train(train_features, train_labels)
    log_reg.test(test_features, test_labels)
    #log_reg.visualize()
    
    # Support Vector Machine
    svm = SupportVectorMachine()
    svm.train(train_features, train_labels)
    svm.test(test_features, test_labels)
    #svm.visualize()

if __name__ == '__main__':

    # predict just using spatial features
    experiment_1(train_seasons=['2006-2007', '2007-2008', '2008-2009'], test_seasons=['2009-2010'])
    
    # predict using location and position features
    experiment_2(train_seasons=['2006-2007', '2007-2008', '2008-2009'], test_seasons=['2009-2010'])

    # predict using location and position features
    experiment_3(train_seasons=['2006-2007', '2007-2008', '2008-2009'], test_seasons=['2009-2010'])



