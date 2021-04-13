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

# Mean-center data and compute SVD
data_mean_centered = data - np.mean(data, axis=0)
U, Sigma, V = np.linalg.svd(data_mean_centered, full_matrices=False)

# Save image
for i in [0, 1, 2]:
    plt.imshow(V.T[:, i].reshape(28, 28), cmap="gray")
    plt.savefig('Plots/Fig_1b_principal_component_' + str(i+1) + '.png')

################################################
# Problem 1c
################################################
print('-----------------------------------------------------------------------')
print('Project data onto principle components...')

# Compute scores
num_components = 2
scores = data_mean_centered @ V.T[:, :num_components]

# Plot data
fig, _ = plt.subplots()
plt.plot(scores[:, 0], scores[:, 1], 'o', markersize=2)
plt.title('Principle Components')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
fig.savefig('Plots/Fig_1c_top_2_principal_component.png')

# 100 and 101th Principle Component
num_components = [99, 100]
scores = data_mean_centered @ V.T[:, 99:102]

# Plot data
fig, _ = plt.subplots()
plt.plot(scores[:, 0], scores[:, 1], 'o', markersize=2)
plt.title('Principle Components')
plt.xlabel('Principle Component 100')
plt.ylabel('Principle Component 101')
fig.savefig('Plots/Fig_1c_100_101_principal_component.png')


################################################
# Problem 1d
################################################
print('-----------------------------------------------------------------------')
print('Compute fractional reconstruction accuracy...')

# Compute Eigenvalues
sum_eigvals = np.sum(np.square(Sigma))

# Fractional reconstruction accuracy
components = np.arange(data_mean_centered.shape[1])
frac_acc = np.zeros((data_mean_centered.shape[1]), )
for i in components:
    frac_acc[i] = np.sum(np.square(Sigma)[:i+1])

# Plot
fig, _ = plt.subplots()
plt.plot(components, (frac_acc / sum_eigvals) * 100)
plt.yticks((10, 20, 30, 40, 50, 60, 70, 80, 90, 100))
plt.xlim([0, 800])
plt.ylim([0, 100])
plt.title('Fractional Reconstruction Accuracy')
plt.xlabel('No. Principle Components Used')
plt.ylabel('Variance Explained (%)')
fig.savefig('Plots/Fig_1d_fractional_reconstruction_accuracy.png')

# Generate table
RC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rows, cols = (10, 2)
accuracy = [[0 for i in range(cols)] for j in range(rows)]
accuracy[0] = ['Reconstruction Accuracy', 'No. Principal Components Used']
components = np.zeros((len(RC), ), dtype=int)
for i in range(len(RC)):
    ele = np.where(np.logical_or((RC[i] * sum_eigvals) > frac_acc, (RC[i] * sum_eigvals) == frac_acc))
    if ele[0].size == 0:
        ele = 0
        components[i] = ele + 1
        accuracy[i+1] = [str(RC[i] * 100) + '%', components[i]]
    else:
        components[i] = ele[0][-1] + 1
        accuracy[i+1] = [str(RC[i] * 100) + '%', components[i]]

# Save table to text file
open('Plots/Table_1d.txt', 'w').write(tabulate(accuracy))

################################################
# Problem 1e
################################################
print('-----------------------------------------------------------------------')
print('Reconstruct sample 1000, 2000, 3000...')

# Reconstruct image
images = np.zeros((10, 28, 28))
Sigma = np.diag(Sigma)
components = np.append(components, data_mean_centered.shape[1])
for img, sample in enumerate([1000, 2000, 3000]):
    for i in range(len(components)):
        X_tilda = U[:, :components[i]] @ Sigma[:components[i], :components[i]] @ V[:components[i], :]
        X = X_tilda + np.mean(data, axis=0)
        images[i, :, :] = X[sample, :].reshape(28, 28)
    plot_reconstruction(images, 'Reconstruction: Sample ' + str(sample), 'Plots/Fig_1e_reconstruction_sample_' +
                        str(sample))


