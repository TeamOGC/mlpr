"""
Created on Sun Apr 23 11:48:37 2023.

@author: OGC
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import numpy.typing as npt


def logpdf_GAU_ND(X: npt.ArrayLike, mu: npt.ArrayLike, C: npt.ArrayLike) -> np.ndarray:
    """Log Density estimation for multiple sample X.

    Parameters
    ----------
    X : npt.ArrayLike
        (M, N) containting for each column a sample.
    mu : npt.ArrayLike
        (M, 1) column vector containing the mean for each feature.
    C : npt.ArrayLike
        (M, M) covariance matrix.

    Returns
    -------
    float
        log density.

    """
    from math import pi
    
       
    features_count = X.shape[0]
    
    constant_factor = np.log(2*pi) * features_count * -0.5;
    constant_factor -= np.linalg.slogdet(C)[1] / 2
    
    """Precision Matrix"""
    P = np.linalg.inv(C)
    centered_sample = X-mu
    p = np.dot(np.dot(centered_sample.T, P), centered_sample)
    
    product = np.diag(p) # To support broadcasting
    return constant_factor - (product / 2)


def loglikelihood(X: npt.ArrayLike, mu: npt.ArrayLike, C: npt.ArrayLike) -> np.float64:
    """Log likelihood

    Parameters
    ----------
    X : npt.ArrayLike
        (M,N) matrix containing N column sample vectors.
    mu : npt.ArrayLike
        (M, 1) column vector containing the mean of each feature of the samples.
    C : npt.ArrayLike
        (M, M) covariance matrix.

    Returns
    -------
    np.float64
        the sum of the log density computed over the features.

    """
    
    return np.sum(logpdf_GAU_ND(X, mu, C))