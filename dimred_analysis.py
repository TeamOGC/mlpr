from ogc.dimensionality_reduction import PCA, LDA
import numpy as np
from os import makedirs
from ogc import plot, utilities
import matplotlib.pyplot as plt
from project import TRAINING_DATA, TEST_DATA, ROOT_PATH

OUTPUT_PATH = ROOT_PATH + "../images/dimred/"
makedirs(OUTPUT_PATH, exist_ok=True)

DTR, LTR = TRAINING_DATA()


def variance(DTR):
    return utilities.cov(DTR).trace()


# LDA and PCA


# LDA

new_dims = list(range(13))
new_dims.reverse()
initial_variance = variance(DTR)
pca_list = []
for new_dim in new_dims:
    pca, _ = PCA(DTR, new_dim)
    pca_list.append((new_dim, variance(pca)/initial_variance))
# Plot variances
plt.figure()
plt.plot([x[0] for x in pca_list], [x[1] for x in pca_list])
plt.xlabel("Number of dimensions")
plt.ylabel("Fraction of Explained Variance")
plt.title("PCA with different number of dimensions")
plt.xticks([x[0] for x in pca_list])
plt.yticks([i/10 for i in range(11)])
print(pca_list)
plt.grid()
plt.savefig(OUTPUT_PATH + "pca_variance.png")


# Again with z-normalization

DTR, _mean, _stddev = utilities.ZNormalization(DTR)
new_dims = list(range(13))
new_dims.reverse()
initial_variance = variance(DTR)
pca_list = []
for new_dim in new_dims:
    pca, _ = PCA(DTR, new_dim)
    pca_list.append((new_dim, variance(pca)/initial_variance))
# Plot variances
plt.figure()
plt.plot([x[0] for x in pca_list], [x[1] for x in pca_list])
plt.xlabel("Number of dimensions")
plt.ylabel("Fraction of Explained Variance")
plt.title("PCA with different number of dimensions - Z-Normalized")
plt.xticks([x[0] for x in pca_list])
plt.yticks([i/10 for i in range(11)])
plt.grid()
print(pca_list)
plt.savefig(OUTPUT_PATH + "pca_variance_znorm.png")
