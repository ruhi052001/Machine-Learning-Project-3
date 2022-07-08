import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

# Creating the 2-dimensional synthetic data with mean, μ = [0, 0]T and covariance matrix, Σ = [[13,-3], [-3, 5]].
covar = [[13, -3], [-3, 5]]
D = np.random.multivariate_normal(mean=[0, 0], cov=covar, size=1000, check_valid="ignore")
D = pd.DataFrame(D, columns=['A', 'B'])

# Q 2(a)
# scatter plot of the data samples
plt.scatter(D['A'], D['B'])
plt.show()

# Q 2(b)
covar_mat = np.dot(np.transpose(D), D) / 1000
val, vect = np.linalg.eig(covar_mat)  # This gives Eigenvectors and Eigenvalues from the Covariance Matrix
print("eigenvalues are: ")
print(val)
print("eigenvectors are: ")
print(vect)

# scatter plot
plt.scatter(D["A"], D["B"])
plt.quiver(0, 0, vect[0][0], vect[1][0], scale=5, angles='xy')
plt.quiver(0, 0, vect[0][1], vect[1][1], scale=5, angles='xy')
plt.show()

# Q 2(c)
A = np.dot(D, vect)
for i in range(2):
    xx = []
    yy = []
    for d in A:
        xx.append(d[i] * vect[0][i])  # x-coordinate of data projected along the Ith Eigenvector
        yy.append(d[i] * vect[1][i])  # y-coordinate of data projected along the Ith Eigenvector
    plt.scatter(D['A'], D['B'])
    plt.scatter(xx, yy)
    plt.quiver(0, 0, vect[0][0], vect[1][0], scale=6, angles='xy')
    plt.quiver(0, 0, vect[0][1], vect[1][1], scale=6, angles='xy')
    plt.show()

# Q 2(d)
pca = decomposition.PCA(n_components=2)
proj = pca.fit_transform(D)
# This fucntion helps us directly find the data as projected along the first n-most significant eigenvectors where
# n is already specified as a part of the previous statement as n_components.
recon = pca.inverse_transform(proj)
recon = pd.DataFrame(recon, columns=['A', 'B'])
# This function reconstructs the data along original dimensions from the projected data using information about the
# eigenvectors and their directions as a function of the original components.
error = 0
#print(D)
#print(recon)
for i in range(len(D)):
    error += ((D['A'][i] - recon['A'][i]) ** 2) + ((D['B'][i] - recon['B'][i]) ** 2) ** 0.5
print(error)


