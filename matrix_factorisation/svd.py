import numpy as np
from math import sqrt
from scipy.sparse import linalg as sla

def scipy_svd(data, k):
    # Use svd implementation from Scipy
    U, S, VT = sla.svds(data, k)
    # Form diagonal matrix from eigenvalues
    return U, np.diag(S), VT

def custom_svd(data, k):
    # Mulitply data matrix with its transpose
    w = data.dot(data.T)
    # Obtain eigenvalues and U
    eig_vals, U = sla.eigsh(w, k)
    # Square-root eigenvalues and form diagonal matrix sigma (S)
    S = np.diag(np.sqrt(eig_vals))
    # Calculate V based on U and the inverse of sigma
    S_inv = np.linalg.inv(S)
    VT = S_inv.dot(U.T.dot(data))
    VT = np.dot(S_inv, np.dot(U.T, data))
    # Remove imaginary parts and return matrices
    return U.real, S.real, VT.real
