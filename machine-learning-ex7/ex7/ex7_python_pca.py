import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids
from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
from recoverData import recoverData
from drawLine import drawLine
from displayData import displayData

#=============== Part 1: PCA for two dimensional data =============
# mat = spio.loadmat('ex7data1.mat', squeeze_me=True)
# data = mat['X']
# # print(data)
# [m, n] = np.shape(data)
#
# df = pd.DataFrame(data, columns=["x", "y"])
# # print(df)
# sns.scatterplot(x="x",y="y", data = df)
# plt.show()
#
# [X_norm, mu, sigma] = featureNormalize(data)
# # print(X_norm)
# [u, s] = pca(X_norm)
# # print(u)
# # print(s)
# k = 1
# z = projectData(X_norm, u, k)
# # print(z)
#
# X_rec  = recoverData(z, u, k);
# # print(X_rec)
#
#
# df = pd.DataFrame(X_norm, columns=["x", "y"])
# # print(df)
# sns.scatterplot(x="x",y="y", data = df)
# df = pd.DataFrame(X_rec, columns=["x", "y"])
# # print(df)
# sns.scatterplot(x="x",y="y", data = df)
# # a = X_norm.transpose()
# # b = X_rec.transpose()
# #
# # a1 = np.concatenate((a[0], b[0]))
# # b1 = np.concatenate((a[1], b[1]))
# # print(a1)
# for i in range(m):
#     drawLine([X_norm[i][0], X_rec[i][0]], [X_norm[i][1], X_rec[i][1]])
#
# plt.show()


# =============== Part 2: Loading and Visualizing Face Data =============

mat = spio.loadmat('ex7faces.mat', squeeze_me=True)
data = mat['X']
# print(data)
[m, n] = np.shape(data)
w = round(math.sqrt(n))
h = int(n/w)
# print(w)
# print(h)
# # def rgb2gray(rgb):
# #     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
# print(data[0])
# data2 = np.reshape(data[0], (h, w))
# print(data2)
# data2 = np.transpose(data2)
# print(data2)
# # data2 = rgb2gray(data2)
# plt.imshow(data2)
# plt.title('Compressed, with #d colors.')
# plt.show()
# x_modified = []
# temp = np.reshape([1,2,3,4,5,6,7,8, 9], (3,3))
# temp = list(np.transpose(temp).ravel())
# print(temp)
# x_modified.append(temp)
# print(x_modified)
# plt.imshow(x_modified[0])
# plt.title('Compressed, with #d colors.')
# plt.show()

example_width = round(math.sqrt(n))
example_height = int(n / example_width)
data_modified = []
for i in range(m):
    temp = np.reshape(data[i], (example_height, example_width))
    temp = list(np.transpose(temp).ravel())
    data_modified.append(temp)
# plt.imshow(np.reshape(temp, (example_height, example_width)))
# plt.show()
# print(np.shape(data))
# print(np.shape(data_modified))
displayData(data_modified[0:100], None)
[X_norm, mu, sigma] = featureNormalize(data_modified)

[u, s] = pca(X_norm)
u=np.transpose(u)
displayData(u[:36], None)

k = 100
# print('----------------------')
# print(m)
# print(n)
# print(np.shape(u))
# print(np.shape(u[:36]))
z = projectData(X_norm, u, k)
X_rec = recoverData(z, u, k)
print(np.shape(data))
print(np.shape(X_rec))
displayData(X_rec[:100], None)


# =============== Optional (ungraded) Exercise: PCA for Visualization =============
# print('Applying K-Means to compress an image.')
# fname = "bird_small.png"
# image1 = np.array(plt.imread(fname))
# image = image1/255
# image_size = image.size
# # print(image)
# # print(image_size)
# # print(image.shape)
# x = image.shape[0]
# y = image.shape[1]
# image_formatted = np.reshape(image, (x*y, 3))
# # print(image)
#
# k = 16;
# max_iters = 10;
#
# initial_centroids = kMeansInitCentroids(image_formatted, k);
# print(initial_centroids)
#
# # Run K-Means
# [centroids, idx] = runkMeans(image_formatted, initial_centroids, max_iters, False);
#
# idx = findClosestCentroids(image_formatted, centroids);
#
# feature_size = len(image_formatted[0])
# len_idx = len(idx)
# X_recovered =  [[0]*feature_size for _ in range(len_idx)]
#
# for i in range(len_idx):
#     for j in range(feature_size):
#         X_recovered[i][j] = centroids[idx[i]-1][j]
#
# X_recovered = np.reshape(X_recovered, (x, y, 3));
# X_recovered = X_recovered * 255
#
# plt.subplot(1, 2, 1);
# plt.imshow(image1)
# plt.title('Original');
# plt.show()
#
# plt.subplot(1, 2, 2);
# plt.imshow(X_recovered)
# plt.title('Compressed, with #d colors.');
# plt.show()