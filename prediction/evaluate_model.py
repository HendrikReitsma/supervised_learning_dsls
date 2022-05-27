import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate_model(model, x_train, y_train, x_test, y_test): 
    """
    Input: model to evaluate, training data and test data.
    :return: mean accuracy of 10 runs, a classificiation report and a confusion matrix
    """   
    # Train and predict 10 times to evaluate time and accuracy
    predictions = []
    run_times = []
    accuracy_scores = []
    
    for _ in range(10):
        start_time = time.time()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions.append(y_pred)
    
        end_time = time.time()
        run_times.append(end_time - start_time)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    # Run time and predictions need to be averaged
    run_time = np.mean(run_times)
    predictions = np.mean(np.array(predictions), axis = 0)
    
    # Calculate performance metrics
    errors = abs(predictions - y_test)
    mean_error = np.mean(errors)
    mean_accuracy = np.mean(accuracy_scores)
    
    # Return results in a dictionary
    results = {'time': run_time, 'error': mean_error, 'mean_accuracy': mean_accuracy}
    
    # Show confusion matrix
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return results