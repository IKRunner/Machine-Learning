# # Gaussian Mixture Models


import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.stats import multivariate_normal
from gmm import GaussianMixtureModel


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
    for i, learned_model in enumerate(learned_models):
        plot_gmm_model(axes[i], learned_model, X_test_all, percentage_data[i])

    axes[-1].axis('off')
    axes[-2].axis('off')
    return fig


## Your code starts here
print("hello world")
