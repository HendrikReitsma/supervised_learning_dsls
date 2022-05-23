from sklearn.ensemble import RandomForestClassifier


def random_forest(X_train, y_train):
  classifier = RandomForestClassifier(max_depth=2)
  classifier.fit(X_train, y_train)
  return classifier
  
