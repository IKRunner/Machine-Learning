# # Principal component analysis

import numpy as np
import matplotlib.pyplot as plt
import utils
from numpy import genfromtxt



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
# plt.show()

################################################
# Problem 1b
################################################
print('-----------------------------------------------------------------------')
print('Run PCA on data...')

# Mean center data
data_mean_centered = data - np.mean(data, axis=0)

# Generate covariance matrix
m = data_mean_centered.shape[0]
cov_matrix = (1 / (m - 1)) * (data_mean_centered.T @ data_mean_centered)

# Compute Eigenvectors and Eigenvalues
eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

# Sort Eigenvalues and corresponding Eigenvectors in descending order
num_components = 1
sorted_idx = np.argsort(eig_values)[::-1]
sorted_eigvalues = eig_values[sorted_idx]
sorted_eigvectors = eig_vectors[:, sorted_idx]
eigvector_subset = sorted_eigvectors[:, 0:num_components]

# Compute scores
# scores = data_mean_centered @

t=0
