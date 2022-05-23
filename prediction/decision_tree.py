from sklearn.tree import DecisionTreeClassifier


def decision_tree(X_train, y_train):
  classifier = DecisionTreeClassifier(max_leaf_nodes=12, random_state=42)
  classifier.fit(X_train, y_train)
  return classifier


