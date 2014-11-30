NBA_shot_analysis
=================

Code for running various classifiers on NBA shot data.  Dependencies: [Scikit-Learn](http://scikit-learn.org/stable/index.html) (and its dependencies)

First, we need to import data helpers and the learning models
```python
from data.data_accessors import load_seasons
from models.classifiers import GaussianMixtureClassifier
```
Next we setup an experiment by loading data by season
```python
train_seasons = ['2006-2007', '2007-2008']
test_seasons = ['2008-2009']
# load training data
made_data, missed_data = load_seasons(train_seasons, split_flag=True)
# load test data
test_features, test_labels = load_seasons(test_seasons)
```

Then we can train one of several models.  A mixture of Gaussians, for instance...
```python
# Mixture of 2-D Gaussians Classifier
num_of_mixtures = 3
gauss_mixture_model = GaussianMixtureClassifier(num_of_mixtures)
gauss_mixture_model.train(made_data, missed_data, max_iterations=10)
gauss_mixture_model.test(test_features, test_labels)
gauss_mixture_model.visualize()
```
Training progress, error on the test set, and a visualization of the decision boundaries will be output:
```
Log-Likelihood Progress:
Iteration #0: -2293316.4711
Iteration #1: -2293292.3619
Iteration #2: -2293155.8165
Iteration #3: -2292613.3473
Iteration #4: -2291197.9684
Iteration #5: -2289138.5734
Iteration #6: -2287083.3789
Iteration #7: -2284983.8981
Iteration #8: -2282518.7470
Iteration #9: -2278953.1199
Training complete.

Made Gauss #0 Params: mu=[25.24,12.32], cov=[108.21,-0.35;-0.35,63.16]
Missed Gauss #0 Params: mu=[25.36,10.98], cov=[144.35,0.23;0.23,38.72]

Made Gauss #1 Params: mu=[25.09,10.56], cov=[94.97,-0.86;-0.86,45.38]
Missed Gauss #1 Params: mu=[25.02,19.46], cov=[135.78,-3.60;-3.60,112.61]

Made Gauss #2 Params: mu=[25.06,13.61], cov=[114.33,-2.03;-2.03,74.53]
Missed Gauss #2 Params: mu=[25.28,13.89], cov=[147.11,-1.21;-1.21,62.94]

Test Accuracy: 0.609823
```
Decision Boundaries (Green = Make, Red = Miss)
![Alt text](/results/spatial_features_results/Gaussian_Mixture_Model.png?raw=true "Decision Boundary for Mix. of Gaussians")
