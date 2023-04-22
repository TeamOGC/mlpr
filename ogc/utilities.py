# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:53:18 2023

@author: alex_
"""
import numpy as np
import numpy.typing as npt


def vcol(arr: "npt.ArrayLike") -> np.ndarray:
    """
    Transorms a numpy array into a column vector

    Parameters
    ----------
    arr : npt.ArrayLike
        input array.

    Returns
    -------
    np.ndarray
        column vector with (arr.size, 1) shape.

    """
    return arr.reshape((arr.size, 1))


def vrow(arr: "npt.ArrayLike") -> np.ndarray:
    """
    Transorms a numpy array into a row vector

    Parameters
    ----------
    arr : npt.ArrayLike
        input array.

    Returns
    -------
    np.ndarray
        row vector with (1, arr.size) shape.

    """
    
    return arr.reshape((1, arr.size))


def cov(data: "npt.ArrayLike") -> np.ndarray:
    """
    Covariance matrix of the input data

    Parameters
    ----------
    data : npt.ArrayLike
        Input matrix.

    Returns
    -------
    np.ndarray
        Covariance Matrix.

    """
    
    mu = data.mean(1)
    data_centered = data - vcol(mu)
    return 1/data.shape[1] * np.dot(data_centered, data_centered.T)
