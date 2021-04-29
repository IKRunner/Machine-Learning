import numpy as np


# Log likelihood expression
def llh(tp1, t1p1, t2p1, t1m1, t2m1):
    return 2 * np.log(tp1 * t1p1 * t2p1) + \
           np.log(tp1 * t1p1 * (1 - t2p1)) + \
           np.log(tp1 * (1 - t1p1) * (1 - t2p1)) + \
           np.log(tp1 * t1m1 * (1 - t2m1)) + \
           np.log(tp1 * (1 - t1m1) * t2m1) + \
           2 * np.log(tp1 * (1 - t1m1) * (1 - t2m1)) + \
           2 * np.log(tp1 * t1p1 * t2p1 + tp1 * t1m1 * t2m1) + \
           2 * np.log(tp1 * (1 - t1p1)*(1 - t2p1) + tp1 * (1 - t1m1) * (1 - t2m1))


# Initial parameters
init_theta_plus1 = 0.5
init_theta1_plus1 = 0.75
init_theta2_plus1 = 0.5
init_theta1_minus1 = 0.25
init_theta2_minus1 = 0.25

val = llh(init_theta_plus1, init_theta1_plus1, init_theta2_plus1, init_theta1_minus1, init_theta2_minus1)

print("Log-likelihood under initial parameter estimates is: " + str(val))


# Updated parameters
theta_plus1 = 0.5065
theta1_plus1 = 0.7756
theta2_plus1 = 0.6111
theta1_minus1 = 0.2171
theta2_minus1 = 0.2171

val = llh(theta_plus1, theta1_plus1, theta2_plus1, theta1_minus1, theta2_minus1)

print("Log-likelihood under updated parameter estimates is: " + str(val))
