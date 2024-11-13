import numpy as np

# Refer notes to see why binary cross entropy is used and how we arrived at these formulas
def binary_cross_entropy(y_true, y_pred):
    return np.mean ( -y_true * np.log(y_pred) - (1-y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - (y_true / y_pred)) / np.size(y_true)