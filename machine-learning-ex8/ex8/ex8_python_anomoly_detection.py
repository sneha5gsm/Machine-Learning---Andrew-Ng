import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian

mat = spio.loadmat('ex8data1.mat', squeeze_me=True)
# print(mat.keys())
data = mat['X']
xval = mat['Xval']
yval = mat['yval']

df = pd.DataFrame(data, columns=["Latency (ms)", "Throughput (mb/s)"])
sns.scatterplot(x="Latency (ms)", y="Throughput (mb/s)", data=df)

plt.show()

[mu, sigma2] = estimateGaussian(data)
# print(mu)
print(np.shape(sigma2))
print(sigma2)
p = multivariateGaussian(data, mu, sigma2)
