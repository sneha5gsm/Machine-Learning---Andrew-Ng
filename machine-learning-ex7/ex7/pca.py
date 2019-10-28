# function [U, S] = pca(X)
#PCA Run principal component analysis on the dataset X
#   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
#   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
#
#
# Useful values
# [m, n] = size(X);
#
# You need to return the following variables correctly.
# U = zeros(n);
# S = zeros(n);
#
# ====================== YOUR CODE HERE ======================
# Instructions: You should first compute the covariance matrix. Then, you
#               should use the "svd" function to compute the eigenvectors
#               and eigenvalues of the covariance matrix.
#
# Note: When computing the covariance matrix, remember to divide by m (the
#       number of examples).
import numpy as np
def pca(data):
    data2 = np.array(data)
    data2_shape = np.shape(data2)
    m=data2_shape[0]
    data2_transpose = np.transpose(data2)
    covariance = np.matmul(data2_transpose, data2)
    covariance = covariance/m
    [u, s, x] = np.linalg.svd(covariance)
    return u, s

