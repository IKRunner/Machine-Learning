# # Viterbi algorithm


import numpy as np


def viterbi(pi, A, phi, X):
    """ Viterbi algorithm for finding MAP estimates for 
    Hidden Markov Model

    NOTE: Feel free to modify this function

    Args:
        pi (numpy.array): Initial distribution

        A (numpy.array): State transition probabilities

        phi (numpy.array): Emission probabilities

        X (numpy.array): Observation sequence

    Returns:
        (Z, prob)
            Z (list): MAP estimate of states
            prob (float): joint distribution (state, observation)

    """
    I = A.shape[0]
    N = len(X)
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(pi + tiny)
    B_log = np.log(phi + tiny)


    D_log = np.zeros((I, N))
    E = np.zeros((I, N - 1)).astype(np.int32)
    D_log[:, 0] = C_log + B_log[:, 0]


    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n - 1]
            D_log[i, n] = np.max(temp_sum) + B_log[i, int(X[n])]
            E[i, n - 1] = np.argmax(temp_sum)


    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N - 2, 0, -1):
        S_opt[n] = E[int(S_opt[n + 1]), n]

    return S_opt, D_log, E

    return Z, prob


# Initialize observation sequences
X = np.zeros((3, 10))
# sing = 0, walk = 1, tv = 2
X[0] = 0, 1, 2, 0, 1, 2, 0, 1, 2, 0
X[1] = 0, 0, 0, 2, 2, 2, 2, 2, 2, 2
X[2] = 2, 1, 2, 0, 0, 0, 1, 2, 0, 0

# Initialize hidden states
# states = np.array(['happy', 'sad'])

# Initialize start probabilities
pi = np.zeros((2, ))
pi = [0.5, 0.5]

# Initialize state transition matrix
A = np.array([[0.8, 0.2], [0.5, 0.5]])
phi = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
S, D, E = viterbi(pi, A, phi, X[0])
print('exp(D_log) =', np.exp(D), sep='\n')
S, D, E = viterbi(pi, A, phi, X[1])
print('exp(D_log) =', np.exp(D), sep='\n')
S, D, E = viterbi(pi, A, phi, X[2])
print('exp(D_log) =', np.exp(D), sep='\n')

t=1
