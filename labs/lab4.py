"""
Created on Sun Apr 23 12:08:30 2023.

@author: alex_
"""
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



import matplotlib.pyplot as plt
import numpy
from ogc.density_estimation import logpdf_GAU_ND, loglikelihood
from ogc.utilities import vrow, cov


XPlot = numpy.linspace(-8, 12, 1000)
m = numpy.ones((1,1)) * 1.0
C = numpy.ones((1,1)) * 2.0
r = logpdf_GAU_ND(vrow(XPlot), m, C)
plt.figure()
plt.plot(XPlot.ravel(), numpy.exp(r))
plt.show()




plt.figure()
X1D = numpy.load("loads/X1D.npy")
plt.hist(X1D.ravel(), bins=100, density=True)
m_ML = X1D.mean(1)
C_ML = cov(X1D)
plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
ll = loglikelihood(X1D, m_ML, C_ML)