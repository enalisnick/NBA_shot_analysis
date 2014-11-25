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
gauss_mixture_model.train(made_data, missed_data, 10)
gauss_mixture_model.test(test_features, test_labels)
gauss_mixture_model.visualize()
```

![Alt text](NBA_shot_analysis/position_graphs/G/Gaussian_Mixture_Model.png?raw=true "Decision Boundary for Mix. of Gaussians")
