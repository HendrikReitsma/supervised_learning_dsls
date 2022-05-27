from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

class ClassifierTuning:
  def __init__(self, X_train, y_train, random_state=42):
    self.X_train = X_train
    self.y_train = y_train
    self.random_state = random_state
    
  def tuning_random(self, classifier, random_grid):
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = classifier,
                                    param_distributions = random_grid,
                                    n_iter = 100, cv = 3, verbose=2,
                                    random_state=42, n_jobs = -1)
    rf_random.fit(self.X_train, self.y_train)
    return rf_random

  def tuning_grid(self, classifier, param_grid, scoring):
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid,
                              cv = 3, n_jobs = -1, verbose = 2, scoring = scoring)
    grid_search.fit(self.X_train, self.y_train)
    return grid_search