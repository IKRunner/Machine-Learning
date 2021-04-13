# # Principal component analysis

import numpy as np
import matplotlib.pyplot as plt
import utils
from numpy import genfromtxt
from tabulate import tabulate


from sklearn.datasets import load_digits
from sklearn.decomposition import TruncatedSVD
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.ensemble import RandomForestClassifier


def plot_reconstruction(images, title, filename):
    '''
    Plots 10%, 20%, ..., 100% reconstructions of a 28x28 image

    Args
        images (numpy.array)
            images has size (10, 28, 28)
        title (str)
            title within the image
        filename (str)
            name of the file where the image is saved

    Returns
        None

    Example usage:
        >>> images = np.zeros(10,28,28)
        >>> images[0,:,:] = x10.reshape((28,28))
        >>> images[1,:,:] = x20.reshape((28,28))
        >>> ...
        >>> images[9,:,:] = x100.reshape((28,28))
        >>> utils.plot_reconstruction(images, 'Image Title', 'filename.png')
    '''
    assert images.shape == (10,28,28)
    fig, (
        (ax0, ax1, ax2, ax3),
        (ax4, ax5, ax6, ax7),
        (ax8, ax9, _, _)
    ) = plt.subplots(3, 4)
    axes = [ax9, ax8, ax7, ax6, ax5, ax4, ax3, ax2, ax1, ax0]
    percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in range(10):
        ax = axes[i]
        percent_name = f'{percents[i]}%' if i != 9 else 'Original'
        ax.set(title=percent_name)
        axes[i].imshow(images[i,:,:], cmap='gray')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)


################################################
# Problem 1a
################################################
print('-----------------------------------------------------------------------')
print('Generate last example...')

# Load data and display
data = genfromtxt('P1/X_train.csv', delimiter=',')
last_example = np.reshape(data[-1, :], (28, 28))
plt.imshow(last_example, cmap="gray")

# Save image
plt.savefig('Plots/Fig_1a_last_example.png')

################################################
# Problem 1b
################################################
print('-----------------------------------------------------------------------')
print('Run PCA on data using SVD...')

# Standardize data
data_standardized = data - np.mean(data, axis=0)

# Compute SVD
U, Sigma, V = np.linalg.svd(data_standardized, full_matrices=False)

# Save image
for i in [0, 1, 2]:
    plt.imshow(V.T[:, i].reshape(28, 28), cmap="gray")
    plt.savefig('Plots/Fig_1b_principal_component_' + str(i+1) + '.png')

################################################
# Problem 1c
################################################
print('-----------------------------------------------------------------------')
print('Project data onto principle components...')

# Mean-center data and compute scores
data_mean_centered = data - np.mean(data, axis=0)
num_components = 2
scores = data_mean_centered @ V.T[:, :num_components]

# Plot data
fig, ax1 = plt.subplots()
ax1.plot(scores[:, 0], scores[:, 1], 'o', markersize=2)
plt.title('Principle Components')
ax1.set_xlabel('Principle Component 1')
ax1.set_ylabel('Principle Component 2')
plt.savefig('Plots/Fig_1c_top_2_principal_component.png')
# plt.close(fig)

# 100 and 101th Principle Component
num_components = [99, 100]
scores = data_mean_centered @ V.T[:, 99:102]

# Plot data
fig, ax = plt.subplots()
ax.plot(scores[:, 0], scores[:, 1], 'o', markersize=2)
plt.title('Principle Components')
ax.set_xlabel('Principle Component 100')
ax.set_ylabel('Principle Component 101')
plt.savefig('Plots/Fig_1c_100_101_principal_component.png')
# plt.close(fig)


################################################
# Problem 1d
################################################
print('-----------------------------------------------------------------------')
print('Compute fractional reconstruction accuracy...')

# Generate covariance matrix
m = data_standardized.shape[0]
cov_matrix = (1 / (m - 1)) * (data_standardized.T @ data_standardized)

# Compute Eigenvectors and Eigenvalues
eig_values, eig_vectors = np.linalg.eig(cov_matrix)
sum_eigvals = np.sum(eig_values)

# Sort Eigenvalues and corresponding Eigenvectors in descending order
sorted_idx = np.argsort(eig_values)[::-1]
sorted_eigvalues = eig_values[sorted_idx]

# Fractional reconstruction accuracy
components = np.arange(data_standardized.shape[1])
frac_acc = np.zeros((data_standardized.shape[1]), )
for i in components:
    frac_acc[i] = np.sum(sorted_eigvalues[:i+1])

# Plot
fig, ax = plt.subplots()
ax.plot(components, (frac_acc / sum_eigvals) * 100)
ax.set_yticks((10, 20, 30, 40, 50, 60, 70, 80, 90, 100))
ax.set_xlim([0, 800])
ax.set_ylim([0, 100])
plt.title('Fractional Reconstruction Accuracy')
ax.set_xlabel('No. Principle Components Used')
ax.set_ylabel('Variance Explained (%)')
plt.savefig('Plots/Fig_1d_fractional_reconstruction_accuracy.png')

# Generate table
RC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rows, cols = (10, 2)
data = [[0 for i in range(cols)] for j in range(rows)]
data[0] = ['Reconstruction Accuracy', 'No. Principal Components Used']
for i in range(len(RC)):
    RC[i] * sum_eigvals
    data[i+1] = [str(RC[i]), 1123]
    # data = [['Reconstruction Accuracy','No. Principal Components Used '],
    #     ["Himanshu",1123],
    #     ["Rohit",1126],
    #     ["Sha",111178]]

print(tabulate(data, headers='firstrow'))

open('Plots/Table_1d.txt', 'w').write(tabulate(data))

sorted_eigvectors = eig_vectors[:, sorted_idx]
# eigvector_subset = sorted_eigvectors[:, 0:num_components]

# plt.imshow(sorted_eigvectors[:, 0].reshape(28, 28), cmap="gray")
# plt.show()



####### Hre we goo
# X, y = load_digits(return_X_y=True)
# image = X[0]
# image = image.reshape((8, 8))
# plt.matshow(image, cmap = 'gray')
# U, s, V = np.linalg.svd(image)
# S = np.zeros((image.shape[0], image.shape[1]))
# S[:image.shape[0], :image.shape[0]] = np.diag(s)
# n_component = 2
# S = S[:, :n_component]
# VT = V.T[:n_component, :]
# A = U.dot(S.dot(VT))
# plt.matshow(A, cmap = 'gray')
######

# Compute SVD
# Sigma = np.diag(Sigma)
# num_compoonents = 10
# # X = U[:, :num_compoonents] @ Sigma[:num_compoonents, :num_compoonents] @ V_T[:num_compoonents, :]

# X += np.mean(data, axis=0)

# last_example = np.reshape(X[-1, :], (28, 28))
# plt.imshow(last_example, cmap="gray")
# plt.show()

# w,v = np.linalg.eig(data_mean.T @ data_mean)

# Compute scores
# scores = data_mean_centered @

t=0
