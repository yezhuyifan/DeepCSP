import autograd.numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to calculate the root mean square error
def root_mean_squared_error(X, Y):
    return np.sqrt(mean_squared_error(Y, X))