NBA_shot_analysis
=================

Code for running various classifiers on NBA shot data.

We setup an experiment by loading data by season
```python
train_seasons = ['2006-2007', '2007-2008']
test_seasons = ['2008-2009']
# load training data
made_data, missed_data = load_seasons(train_seasons, split_flag=True)
train_features, train_labels = load_seasons(train_seasons)
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
Iteration #0: -1174166.7747
Iteration #1: -1174152.5590
Iteration #2: -1174082.9130
Iteration #3: -1173812.6065
Iteration #4: -1173228.5472
Iteration #5: -1172640.0817
Iteration #6: -1172203.9383
Iteration #7: -1171873.9845
Iteration #8: -1171618.4053
Iteration #9: -1171415.8183
Training complete.

Made Gauss #0 Params: mu=[25.18,13.83], cov=[131.24,-1.52;-1.52,73.60]
Missed Gauss #0 Params: mu=[25.37,13.84], cov=[169.41,-0.91;-0.91,63.87]

Made Gauss #1 Params: mu=[25.19,13.60], cov=[130.70,-1.38;-1.38,71.89]
Missed Gauss #1 Params: mu=[25.37,14.75], cov=[167.21,-1.40;-1.40,69.49]

Made Gauss #2 Params: mu=[25.30,13.55], cov=[130.75,-0.85;-0.85,71.44]
Missed Gauss #2 Params: mu=[25.14,19.72], cov=[148.56,-3.55;-3.55,122.63]

Test Accuracy: 0.580941
```
Decision Boundaries (Green = Make, Red = Miss)
![Alt text](/position_graphs/G/Gaussian_Mixture_Model.png?raw=true "Decision Boundary for Mix. of Gaussians")
