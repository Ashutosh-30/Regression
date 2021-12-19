
import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################

    # mse = mean((y-y')^2) -> by definiton
    y_dash = np.dot(X, w) if (np.size(X ,1 )==np.size(w ,0)) else np.dot(X.T, w) # calculate the prediction y', had to check the dimensionailty before taking transpose
    # print(y_dash)
    # print(np.mean((y - y_dash)**2))
    return np.mean((y - y_dash )**2)

###### Part 1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here                    #
    #####################################################
    # X_T_X_I = np.linalg.inv(np.dot(X.T, X))
    # print(X_T_X_I)
    # X_T_Y = np.dot(X.T, y)
    # print(X_T_Y)
    # print(np.dot(X_T_X_I, X_T_Y))
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here                    #
    #####################################################
    # print(lambd * np.identity(np.size(X, 1)))
    # print(np.dot(np.linalg.inv(np.add(X_T_X, lambda_I)), X_T_Y))
    return np.dot(np.linalg.inv(np.add(np.dot(X.T, X), lambd * np.identity(np.size(X, 1)))), np.dot(X.T, y))

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################
    result = None
    min = float("inf")
    for p in range(-14, 1):
        l = 2 ** p # the lambda
        w = regularized_linear_regression(Xtrain, ytrain, l)
        err = mean_square_error(w, Xval, yval)
        if err < min: # if better performance, change variables
            result = l
            min = err
    return result


###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################
    temp = X
    for p in range(2, p+ 1):
        X = np.concatenate((X, np.power(temp, p)), axis=1)
    return X


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""




