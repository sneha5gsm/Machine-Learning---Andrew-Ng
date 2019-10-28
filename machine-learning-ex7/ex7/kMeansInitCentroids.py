# function centroids = kMeansInitCentroids(X, K)
# %KMEANSINITCENTROIDS This function initializes K centroids that are to be
# %used in K-Means on the dataset X
# %   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
# %   used with the K-Means on the dataset X
# %
#
# % You should return this values correctly
# centroids = zeros(K, size(X, 2));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: You should set centroids to randomly chosen examples from
# %               the dataset X
import numpy as np
def kMeansInitCentroids(data, k):
    features = len(data[0])
    centroids = [[0]*features for _ in range(k)]
    limit = len(data)
    numbers = np.random.randint(0, limit-1, size=(1,k))
    # print(numbers)
    i=0
    for x in np.nditer(numbers):
        for j in range(features):
            centroids[i][j] = data[x][j]
        i += 1
    return centroids

# print(kMeansInitCentroids([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5]], 2))
