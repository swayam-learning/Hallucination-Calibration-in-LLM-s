import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (y_true == y_pred).mean()

def brier_score(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return np.mean((y_prob - y_true) ** 2)
