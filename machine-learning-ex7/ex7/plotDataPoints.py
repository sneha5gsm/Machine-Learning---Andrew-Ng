# function plotDataPoints(X, idx, K)
#PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
#index assignments in idx have the same color
#   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
#   with the same index assignments in idx have the same color

# Create palette
# palette = hsv(K + 1);
# colors = palette(idx, :);
#
# # Plot the data
# scatter(X(:,1), X(:,2), 15, colors);
#
# end

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotDataPoints( data, idx, k):
    # palette = sns.color_palette("hsv", n_colors=k+1)
    arr1 = np.array(idx)
    arr2 = np.array(data)
    arr1 = np.reshape(arr1, (-1, 1))
    data_new = np.concatenate((arr2, arr1), axis=1)
    df = pd.DataFrame(data_new, columns=["x", "y", "z"])
    # print(df)
    sns.scatterplot(x="x",y="y", hue="z", data = df)
    plt.show()

# plotDataPoints([[1,2],[3,4], [5,6]], [7, 8, 7], 2)
