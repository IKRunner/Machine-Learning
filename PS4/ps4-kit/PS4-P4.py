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
    '''
    # Your code goes here
    t observation variables
    # t hidden variables
    # hmm defines joint prob distr over all 2t variables
    # hgidden states one of k possible values
    each variable x takes one of m possible values
    Conditional indepence assumption
        conditonally independent if joint dist is equal to product of individual dist of x and y
    p(x|y, z) = p(x|z)
    
    Hmm
    Given hidden state at some time/posiiton t, mext hidden state (z_{t+1})is conditionally independent of all previous hidden 
    states and all observations up to t
    Same with obersation at time t and t+1
    Jopint distribution over all random variables has a unique structure (slide 13)
    Conditional probabilities will simplify to slide 14
    need conditional distribution of initial state!!!!
    Initial state probabilities (kx1 vector) probablility that first hidden state takes values k (\pi_k)
    state transition probabilities probability of being in next hidden state is A_{jk} (kxk matrix). In current state ja, 
    what is probability that next hidden staet is k? Each row of matrix adds to one
    Observation/emmission probailities, given that in hidden state k, what is probability that produce particular observation x?
    Can plug parameters into joint probability distribution (slide 16)
    How to estimate parameters given data?
    For each t, compute posterior distirbution of z_t
    '''

    return Z, prob

