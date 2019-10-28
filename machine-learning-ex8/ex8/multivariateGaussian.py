# function p = multivariateGaussian(X, mu, Sigma2)
#MULTIVARIATEGAUSSIAN Computes the probability density function of the
#multivariate gaussian distribution.
#    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability
#    density function of the examples X under the multivariate gaussian
#    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
#    treated as the covariance matrix. If Sigma2 is a vector, it is treated
#    as the \sigma^2 values of the variances in each dimension (a diagonal
#    covariance matrix)
#

# k = length(mu);
#
# if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
#     Sigma2 = diag(Sigma2);
# end
#
# X = bsxfun(@minus, X, mu(:)');
# p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
#     exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));
#
# end

import math
import numpy as np
def multivariateGaussian(x, mu, sigma2):
    k = len(mu)
    x = x - mu
    # p = math.pow((2 * math.pi * sigma2), -0.5) * (math.pow(math.e, (0 - math.pow((x - mu), 2))/(2 * math.pow(sigma2, 2))))
    # print(np.finfo(np.double).precision)
    # sigma2 = sigma2[1:]
    # sigma2 = np.transpose(sigma2)
    # sigma2 = sigma2[1:]
    # sigma2=np.transpose(sigma2)
    (sign, logdet) = np.linalg.slogdet(sigma2)
    print(logdet)
    sigma2_det = logdet * sign
    # print(np.shape(sigma2))
    print('sigma2 det')
    print(sigma2_det)
    p = math.pow((2 * math.pi), (-k/2)) * sign
    y = np.matmul(np.transpose(x), np.linalg.inv(sigma2))
    print(y)
    print(np.shape(x))
    print(np.shape(y))
    z = np.matmul(y, x)
    print(x)
    print(z)
    # print((-0.5 * z) + (sigma2_det * 0.5))
    print((-0.5 * z) - (sigma2_det * 0.5))
    p = p * np.exp((-0.5 * z) - (sigma2_det * 0.5))
    return p
