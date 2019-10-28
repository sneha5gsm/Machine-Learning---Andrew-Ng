# function [X_norm, mu, sigma] = featureNormalize(X)
#FEATURENORMALIZE Normalizes the features in X
#   FEATURENORMALIZE(X) returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation
#   is 1. This is often a good preprocessing step to do when
#   working with learning algorithms.

# mu = mean(X);
# X_norm = bsxfun(@minus, X, mu);
#
# sigma = std(X_norm);
# X_norm = bsxfun(@rdivide, X_norm, sigma);


# ============================================================

# end

import numpy as np
def featureNormalize(raw_data):
    mu = np.mean(raw_data, axis = 0)
    data_norm = raw_data - mu
    sigma = np.std(data_norm, axis=0)
    data_norm = data_norm/sigma
    return data_norm, mu, sigma
