import numpy as np
from os import makedirs
from ogc import plot, utilities
import matplotlib.pyplot as plt
from project import TRAINING_DATA, TEST_DATA, ROOT_PATH
OUTPUT_PATH = ROOT_PATH + "../images/introductory/"
makedirs(OUTPUT_PATH, exist_ok=True)

D, L = TRAINING_DATA()
T1, T2 = TEST_DATA()
# plot.plotFeatures(D, L, "", ["Male", "Female"])
# plot.heatmap(D, L)
# plt.show()
