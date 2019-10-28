# print('Finding closest centroids.\n\n');
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids

mat = spio.loadmat('ex7data2.mat', squeeze_me=True)
data = mat['X']

# print(data)
# Load an example dataset that we will be using
# load('ex7data2.mat');


##################### excercise 1
# Select an initial set of centroids
# k = 3; # 3 Centroids
# centroids = [[3, 3],[6, 2],[8, 5]]
# idx = findClosestCentroids(data, centroids)
# print(idx)
#
# print(centroids)
# centroids = computeCentroids(data, idx, k);
# print(centroids)


######################### excercise 2  - running k means on a 2 d dataset
# initial_centroids = [[3, 3],[6, 2],[8, 5]]
# max_iters = 10;
# [centroids, idx] = runkMeans(data, initial_centroids, max_iters, True);


############################ excercise 3 - running k means to compress an image
print('Applying K-Means to compress an image.')
fname = "bird_small.png"
image1 = np.array(plt.imread(fname))
image = image1/255
image_size = image.size
# print(image)
# print(image_size)
# print(image.shape)
x = image.shape[0]
y = image.shape[1]
image_formatted = np.reshape(image, (x*y, 3))
# print(image)

k = 16;
max_iters = 10;

initial_centroids = kMeansInitCentroids(image_formatted, k);
print(initial_centroids)

# Run K-Means
[centroids, idx] = runkMeans(image_formatted, initial_centroids, max_iters, False);

idx = findClosestCentroids(image_formatted, centroids);

feature_size = len(image_formatted[0])
len_idx = len(idx)
X_recovered =  [[0]*feature_size for _ in range(len_idx)]

for i in range(len_idx):
    for j in range(feature_size):
        X_recovered[i][j] = centroids[idx[i]-1][j]

X_recovered = np.reshape(X_recovered, (x, y, 3));
X_recovered = X_recovered * 255

plt.subplot(1, 2, 1);
plt.imshow(image1)
plt.title('Original');
plt.show()

plt.subplot(1, 2, 2);
plt.imshow(X_recovered)
plt.title('Compressed, with #d colors.');
plt.show()