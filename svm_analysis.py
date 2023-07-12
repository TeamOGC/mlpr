from typing import Dict
from os import makedirs
import matplotlib.pyplot as plt
# from ogc import dimensionality_reduction as dr
from ogc.classifiers import SVM
from ogc import utilities
import numpy.typing as npt
import numpy as np
from project import TRAINING_DATA, ROOT_PATH
import logging
from project import ZNormalization as znorm_cached
from project import PCA as PCA_Cached
from pprint import pprint
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OUTPUT_PATH = ROOT_PATH + "../images/svm_analysis/"
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/svm_analysis/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)


def linear_svm_callback(option, prior, dimred, dataset_type, C, K):
    assert option == "linear"
    DTR, LTR = TRAINING_DATA()
    if dataset_type == "Z-Norm":
        DTR = znorm_cached()
    if dimred != None:
        DTR = PCA_Cached(dimred)

    model = SVM.LinearSVM(C=C, K=K)
    from ogc.utilities import Kfold
    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold


def poly_svm_callback(option, prior, dimred, dataset_type, c, d, C, K):
    assert option == "polynomial"
    DTR, LTR = TRAINING_DATA()
    if dataset_type == "Z-Norm":
        from ogc import utilities as utils
        DTR = utils.ZNormalization(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]
    model = SVM.PolynomialSVM(c=c, d=d, C=C, epsilon=K**2)
    from ogc.utilities import Kfold
    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold


def rbf_svm_callback(option, prior, dimred, dataset_type, gamma, C, K):
    assert option == "RBF"
    DTR, LTR = TRAINING_DATA()
    if dataset_type == "Z-Norm":
        from ogc import utilities as utils
        DTR = utils.ZNormalization(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]
    model = SVM.RBFSVM(gamma, C, K)
    from ogc.utilities import Kfold
    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold


def main():
    fast_run = True
    options = [("Linear", "linear"),
               ("Polynomial", "polynomial"), ("RBF", "RBF")]
    if fast_run:
        priors = [("$\pi = 0.5$", 0.5), ("$\pi = 0.1$", 0.1),
                  ("$\pi = 0.9$", 0.9)]
        dimred = [("No PCA", None), ("PCA $(m=5)$", 5)]
        dataset_types = [("RAW", None), ("Z-Norm", "Z-Norm")]
        cs = [("$c = 1$", 1), ]
        ds = [("$d = 2$", 2), ]
        gammas = [("$\gamma = 10^2$", 100), ]
        Cs = [("$C = 10^{-1}$", 0.1), ]
        Ks = [("$K = 1$", 1), ]
    else:
        priors = [("$\pi = 0.5$", 0.5), ("$\pi = 0.1$", 0.1),
                  ("$\pi = 0.9$", 0.9)]
        dataset_types = [("RAW", None), ("Z-Norm", "Z-Norm")]
        dimred = [("No PCA", None), ("PCA $(m=5)$", 5)]
        cs = [("$c = 0$", 0), ("$c = 1$", 1), ("$c = 10$", 10),
              ("$c = 20$", 20)]  # Polynomial offset
        ds = [("$d = 2$", 2), ("$d = 3$", 3)]  # Polynomial degree
        gammas = [("$\gamma = 10^-3$", 0.001),
                  ("$\gamma = 1$", 1)]  # RBF gamma
        # Regularization parameter
        Cs = [("$C = 10$", 10), ("$C = 1$", 1),
              ("$C = 10^{-2}$", 0.01),  ("$C = 10^{-4}$", 0.0001)]
        Ks = [("$K = 1$", 1), ("$K = 10$", 10)]  # Kernel offset

    linear_poly_rbf = [False, True, False]

    if linear_poly_rbf[0]:
        _, linear_table = utilities.grid_search(
            linear_svm_callback, [options[0]], priors, dimred, dataset_types, Cs, Ks)

        filename = TABLES_OUTPUT_PATH + "svm_results_linear.csv"
        np.savetxt(filename, linear_table, delimiter=";",
                   fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "C", "K", "MinDCF"]))

    if linear_poly_rbf[1]:
        _, poly_table = utilities.grid_search(
            poly_svm_callback, [options[1]], priors, dimred, dataset_types, cs, ds, Cs, Ks)

        filename = TABLES_OUTPUT_PATH + "svm_results_poly.csv"
        np.savetxt(filename, poly_table, delimiter=";",
                   fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "c", "d", "C", "Epsilon", "MinDCF"]))

    if linear_poly_rbf[2]:
        _, rbf_table = utilities.grid_search(
            rbf_svm_callback, [options[2]], priors, dimred, dataset_types, gammas, Cs, Ks)
        filename = TABLES_OUTPUT_PATH + "svm_results_rbf.csv"
        np.savetxt(filename, rbf_table, delimiter=";",
                   fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "Gamma", "C", "K", "MinDCF"]))

    if all(linear_poly_rbf):
        table = np.vstack([linear_table, poly_table, rbf_table])

        np.savetxt(TABLES_OUTPUT_PATH + "logreg_results.csv", table, delimiter=";",
                   fmt="%s", header=";".join(["Prior", "Lambda", "PCA", "Dataset", "MinDCF"]))


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start} seconds")
