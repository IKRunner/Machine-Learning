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
    # Your code goes here

    return Z, prob

