# # Gaussian Mixture Models


import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.stats import multivariate_normal
from gmm import GaussianMixtureModel
from sklearn.mixture import GaussianMixture


def plot_contour_gaussian(ax, mean, covariance, eps=1e-2):
    """ Plot the contour of a 2d Gaussian distribution with given mean and 
    covariance matrix

    Args:
        ax (matplotlib.axes.Axes):
            Subplot used to plot the contour
        mean (numpy.array):
            Mean of the gaussian distribution
        covariance (numpy.array):
            Covariance matrix of the distribution
        eps:
            The cut off to draw the contour plot. The higher the value, 
            the smaller the contour plot.

    Returns:
        None

    """
    x1_range=np.linspace(-6, 8, 100)
    x2_range=np.linspace(-10, 8, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range, indexing='ij')
    Z      = np.concatenate((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis]), axis=1)
    P      = multivariate_normal.pdf(Z, mean, covariance)
    P[P < eps] = 0
    P      = P.reshape((len(x1_range), len(x2_range)))
    ax.contour(x1_range, x2_range, P.T, colors='black', alpha=0.2)
    

def plot_gmm_model(ax, learned_model, test_data, percent):
    """ Plot the learned GMM and its associated gaussian distribution
    against the test data

    Args:
        ax (matplotlib.axes.Axes):
            Subplot used to plot the contour

        learned_model (GaussianMixtureModel):
            A trained GMM
        
        test_data (numpy.float):
            The testing data

        percent (float):
            The percentage of training data, used to label the subplot

    Returns:
        None

    """
    for k in range(learned_model.K):
        plot_contour_gaussian(ax, learned_model.mus[k, :], learned_model.covariances[k, :, :])
    ax.scatter(test_data[:, 0], test_data[:, 1], alpha=0.5)
    ax.scatter(learned_model.mus[:, 0], learned_model.mus[:, 1], c="r")
    ax.set_ylim(-10, 8)
    ax.set_xlim(-6, 8)
    ax.set_title(f"{percent}%")


def plot_multiple_contour_plots(learned_models):
    """ Plot multiple learned GMMs

    Arg:
        learned_models (list):
            A list of learned models that were trained on 10%,
            20%, 30%, ..., 100% of training data
    
    Returns:
        fig:
            The figure handle which you can use to save the figure

    Example usage:
        >>> learned_models = ... # A list of trained GMMs trained on increasing data
        >>> fig = plot_multiple_contour_plots(learned_models)
        >>> fig.savefig("4(a)(ii).png)

    """
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))

    axes = axes.flatten()
    percentage_data = np.arange(10, 101, 10)
    X_test_all = utils.load_data_from_txt_file("P3/X_test.txt")
    for i, learned_model in enumerate(learned_models):
        plot_gmm_model(axes[i], learned_model, X_test_all, percentage_data[i])

    axes[-1].axis('off')
    axes[-2].axis('off')
    return fig


'''
For ith data point, if kth closter mean is closest to ith data point, then point is assighend to cluster ,  else 0
break ties arbitrarily
initlize mean clusters
update cluster assignments
update cluster means
repeat
-gaussian mixture models provide soft clustering
-\pi_k are mixing coefficients
- Given target number oif Gaussian's k, find parameter estimates
hideen variable is which gaussian points came from
introduce latent variable which of k gaussians point came from
initialize parameters to some estimate
E step
'''

################################################
# Problem 3a_i
################################################
print('-----------------------------------------------------------------------')
print('Generate learning curve...')
learned_models = []

# Load testing data
x_test = utils.load_data_from_txt_file("P3/X_test.txt")
x_train = utils.load_data_from_txt_file("P3/X_test.txt")

# No. features
N, d = x_test.shape

# Initialize target number of Gaussians
K = 3

# Initialize mixing coefficients
mix_coeff = np.zeros((K, ))
mix_coeff[:] = 1/K

# Initialize covariance matrices
covar = np.zeros((K, d, d))
covar[:] = np.eye(d)

# Loop through all permutations of training data
for frac in range(10, 110, 10):
    # Load training data from current permutation
    x_train_perm = utils.load_data_from_txt_file("P3/TrainSubsets/X_train_" + str(frac) + "%.txt")

    # Load mean initializations
    mu = utils.load_data_from_txt_file("P3/MeanInitialization/Part_a/mu_" + str(frac) + "%.txt")

    # Instantiate Gaussian mixture model object
    gmm_mdl = GaussianMixtureModel(K, mu, covar, mix_coeff)

    # Learn gmm model on current permutation of training data
    gmm_mdl.fit(x_train_perm)

    learned_models.append(gmm_mdl)

    # fig, ax = plt.subplots()

    # plot_gmm_model(ax, gmm_mdl, x_test, frac)

    # plt.plot(x_train_perm[:, 0], x_train_perm[:, 1], 'bx')
    # plt.axis('equal')
    # plt.show()
    # gmm = GaussianMixture(n_components=3)
    # gmm.fit(x_train_perm)
    # print(gmm.means_)
    # print('\n')
    # print(gmm.covariances_)
    # print('-----------------------------------------------------------------------')
    # print(gmm_mdl.mus)
    # print('\n')
    # print(gmm_mdl.covariances)

fig = plot_multiple_contour_plots(learned_models)
fig.savefig("Plots/4(a)(ii).png")
print('-----------------------------------------------------------------------')