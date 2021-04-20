# # Gaussian Mixture Models


import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.stats import multivariate_normal
from gmm import GaussianMixtureModel
from tabulate import tabulate


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
    x1_range = np.linspace(-6, 8, 100)
    x2_range = np.linspace(-10, 8, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range, indexing='ij')
    Z = np.concatenate((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis]), axis=1)
    P = multivariate_normal.pdf(Z, mean, covariance)
    P[P < eps] = 0
    P = P.reshape((len(x1_range), len(x2_range)))
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


################################################
# Problem 3a
################################################
print('-----------------------------------------------------------------------')
print('Generate learning curve (K = 3)...')
# Container for all models
learned_models = []

# Container to store log-likelihoods
log_likelihoods = np.zeros([10, 2])

# Load training, testing data
x_train = utils.load_data_from_txt_file("P3/X_train.txt")
x_test = utils.load_data_from_txt_file("P3/X_test.txt")
N, d = x_train.shape

# Initialize target number of Gaussians and training partitions
K = 3
perms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Initialize mixing coefficients
mix_coeff = np.zeros((K,))
mix_coeff[:] = 1 / K

# Initialize covariance matrices
covar = np.zeros((K, d, d))
covar[:] = np.eye(d)

train_l = None
test_l = None
it = None
# Loop through all permutations of training data
for frac, perm in enumerate(perms):
    # Load training data from current permutation
    x_train_perm = utils.load_data_from_txt_file("P3/TrainSubsets/X_train_" + str(perm) + "%.txt")

    # Load mean initializations
    mu = utils.load_data_from_txt_file("P3/MeanInitialization/Part_a/mu_" + str(perm) + "%.txt")

    # Instantiate Gaussian mixture model object
    gmm_mdl = GaussianMixtureModel(K, mu, covar, mix_coeff)

    # Learn gmm model on current permutation of training data
    gmm_mdl.fit(x_train_perm)

    # Store current model
    learned_models.append(gmm_mdl)

    # Train and test log-likelihood
    train_log = gmm_mdl.compute_llh(x_train_perm)
    test_log = gmm_mdl.compute_llh(x_test)

    # Save log-likelihood values for permutation 100
    if perm == 100:
        train_l = train_log
        test_l = test_log
        it = gmm_mdl.it
    print("For permutation " + str(perm) + ", iteration " + str(gmm_mdl.it) +
          ", normalized training log-likelihood is: " + str(round(train_log, 4)) +
          ", normalized test log-likelihood is: " + str(round(test_log, 4)))

    # Save log-likelihoods
    log_likelihoods[frac, 0] = train_log
    log_likelihoods[frac, 1] = test_log

# Generate contour plot
fig = plot_multiple_contour_plots(learned_models)
fig.savefig("Plots/3(a)(ii).png")

print("\nParameters of final model:")
# Generate table of data for 100% permutation
rows, cols = (5, 2)
names = ['K', 'Permutation', 'Iterations', 'Normalized training log-likelihood', 'Normalized test log-likelihood']
values = [str(K), str(100) + '%', str(it), str(round(train_l, 4)), str(round(test_l, 4))]
filename = 'Plots/3(a)(ii).txt'
utils.tabulate_data(rows, cols, names, values, "center", "num", filename, 'w')

# Generate table of means
rows, cols = (3, 2)
names = ['Mean 1', 'Mean 2', 'Mean 3']
values = [str(np.round(learned_models[9].mus[0], 4)),
          str(np.round(learned_models[9].mus[1], 4)),
          str(np.round(learned_models[9].mus[2], 4))]
filename = 'Plots/3(a)(ii)_means.txt'
utils.tabulate_data(rows, cols, names, values, "right", "str", filename, 'w')

# Generate table of covariances
rows, cols = (3, 2)
names = ['Covariance 1', 'Covariance 2', 'Covariance 3']
values = [str(np.round(learned_models[9].covariances[0], 4)),
          str(np.round(learned_models[9].covariances[1], 4)),
          str(np.round(learned_models[9].covariances[2], 4))]
filename = 'Plots/3(a)(ii)_covariances.txt'
utils.tabulate_data(rows, cols, names, values, "right", "str", filename, 'w')

# Generate table of mixing coefficients
rows, cols = (3, 2)
names = ['Mixing coefficient 1', 'Mixing coefficient 2', 'Mixing coefficient 3']
values = [str(np.round(learned_models[9].mixing_coeff[0], 4)),
          str(np.round(learned_models[9].mixing_coeff[1], 4)),
          str(np.round(learned_models[9].mixing_coeff[2], 4))]
filename = 'Plots/3(a)(ii)_mixing_coefficients.txt'
utils.tabulate_data(rows, cols, names, values, "right", "str", filename, 'w')
print('-----------------------------------------------------------------------')

# Plot Log-likelihoods
fig, ax = plt.subplots()
ax.plot(np.linspace(10, 100, num=10), log_likelihoods[:, 0], marker='o', c='b', label='Training Log-Likelihood')
ax.plot(np.linspace(10, 100, num=10), log_likelihoods[:, 1], marker='o', c='r', label='Test Log-Likelihood')
plt.title('GMM Learning Curve')
plt.legend()
plt.xlim(0, 110)
ax.set_xlabel('Training Partition (%)')
ax.set_ylabel('Normalized Log-Likelihood')
fig.savefig("Plots/3(a)(i).png")
plt.show(block=False)
################################################
# Problem 3b
################################################
print('Performing cross-validation...')

# Target Gaussians
gauss = [1, 2, 3, 4, 5]

# Container to store log-likelihoods and cross validation log-likelihood
log_likelihoods = np.zeros([len(gauss), 2])
log_likelihoods_cross = np.zeros([5, len(gauss)])

# Loop through all K values
for idx_K, K in enumerate(gauss):
    # Initialize mixing coefficients
    mix_coeff = np.zeros((K,))
    mix_coeff[:] = 1 / K

    # Initialize covariance matrices
    covar = np.zeros((K, d, d))
    covar[:] = np.eye(d)

    # Load mean initializations
    mu = utils.load_data_from_txt_file("P3/MeanInitialization/Part_b/mu_k_" + str(K) + ".txt")

    # Instantiate Gaussian mixture model object for current K value
    gmm_mdl = GaussianMixtureModel(K, mu, covar, mix_coeff)

    # Learn gmm model on full training data
    gmm_mdl.fit(x_train)

    # Train and test log-likelihood
    train_log = gmm_mdl.compute_llh(x_train)
    test_log = gmm_mdl.compute_llh(x_test)

    # Save log-likelihoods
    log_likelihoods[idx_K, 0] = train_log
    log_likelihoods[idx_K, 1] = test_log

    # Loop through all folds
    for idx_cross, fold in enumerate(gauss):
        # Load training/testing data from current fold
        x_train_fold = utils.load_data_from_txt_file("P3/CrossValidation/X_train_fold" + str(fold) + ".txt")
        x_test_fold = utils.load_data_from_txt_file("P3/CrossValidation/X_test_fold" + str(fold) + ".txt")

        # Learn gmm model on training fold
        gmm_mdl.fit(x_train_fold)

        # Test log-likelihood for respective fold
        test_fold_log = gmm_mdl.compute_llh(x_test_fold)

        # Save cross log-likelihoods
        log_likelihoods_cross[idx_cross, idx_K] = test_fold_log

    print("For K = " + str(K) + ", normalized training log-likelihood is: " + str(round(train_log, 4)) +
          ", normalized test log-likelihood is: " + str(round(test_log, 4)) +
          ", average cross validation log-likelihood is: " + str(round(np.average(log_likelihoods_cross[:, idx_K]), 4)))
# Optimal K value
k_chosen = gauss[np.argmax(np.average(log_likelihoods_cross, axis=0))]
print("Selected value of K is: " + str(k_chosen))

# Generate table
rows, cols = (4, 2)
names = ['Selected value of K:', 'Average cross validation log-likelihood ',
         'Normalized training log-likelihood', 'Normalized test log-likelihood']
values = [str(k_chosen), str(round(np.max(np.average(log_likelihoods_cross, axis=0)), 4)),
          str(round(log_likelihoods[np.argmax(np.average(log_likelihoods_cross, axis=0)), 0], 4)),
          str(round(log_likelihoods[np.argmax(np.average(log_likelihoods_cross, axis=0)), 1], 4))]
filename = 'Plots/3(b).txt'
utils.tabulate_data(rows, cols, names, values, "center", "num", filename, 'w')
print('-----------------------------------------------------------------------')

# Plot Log-likelihoods
fig, ax = plt.subplots()
ax.plot(gauss, log_likelihoods[:, 0], marker='o', c='b', label='Training Log-Likelihood')
ax.plot(gauss, log_likelihoods[:, 1], marker='o', c='r', label='Test Log-Likelihood')
ax.plot(gauss, np.average(log_likelihoods_cross, axis=0), marker='o', c='k',
        label='Cross-Validation Log-Likelihood')
plt.title('GMM Learning Curve')
plt.legend()
plt.xticks(np.arange(1, 6))
ax.set_xlabel('No. Gaussians')
ax.set_ylabel('Normalized Log-Likelihood')
fig.savefig("Plots/3(b).png")
plt.show(block=False)
