"""Utility functions"""
import pennylane as qml
import numpy as np
from tqdm import tqdm
import scipy


def hamming_weight(n):
    """Computes the Hamming weight of the binary representation of a decimal number
    Args
        n (int): An integer
    Returns
        int
    """
    return bin(n).count("1")


def vec_hamming_weight(vec, weight, tol=1e-06):
    """Checks if, when expanded in the computational basis, the non-zero components of a vector
    correspond to basis state with a fixed Hamming weight.
    Args
        vec (Iterable): The vector being checked
        weight (int): The fixed Hamming weight
        tol (float): The tolerance for a vector entry being equal to 0
    Returns
        bool
    """
    out = True
    for count, num in enumerate(vec):
        if num > tol and hamming_weight(count) != weight:
            out = False
    return out

def allowed_vec_val(matrix, weight):
    """Returns the eigenvectors v and corresponding eigenvalues w such that
    vec_hamming_weight(v, weight) = True, where weight is some given Hamming weight.
    Args
        matrix (numpy.array): Input matrix
        weight (int): Fixed Hamming weight
    Returns
        (numpy.array, numpy.array)
    """

    w, v = np.linalg.eig(matrix)
    v = np.transpose(v)

    allowed_vals = []
    allowed_vecs = []

    for c, vec in enumerate(v):
        if vec_hamming_weight(vec, weight):
            allowed_vals.append(w[c])
            allowed_vecs.append(vec)

    return allowed_vecs, allowed_vals