from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


class RandomForestClassifierTuning:
  def __init__(self, X_train, y_train, random_state=42):
    self.X_train = X_train
    self.y_train = y_train
    self.random_state = random_state

  def random_forest_base(self):
    classifier = RandomForestClassifier(random_state=self.random_state)
    classifier.fit(self.X_train, self.y_train)
    return classifier

  def random_forest_tuned_random(self, random_grid):
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    classifier = RandomForestClassifier(random_state=self.random_state)
    rf_random = RandomizedSearchCV(estimator = classifier,
                                    param_distributions = random_grid,
                                    n_iter = 100, cv = 3, verbose=2,
                                    random_state=42, n_jobs = -1)
    rf_random.fit(self.X_train, self.y_train)
    return rf_random

  def random_forest_tuned_grid(self, param_grid):
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(self.X_train, self.y_train)
    return grid_search
  
