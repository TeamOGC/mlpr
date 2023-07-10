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
from pprint import pprint
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OUTPUT_PATH = ROOT_PATH + "../images/svm_analysis/"
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/svm_analysis/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)


def linear_svm_callback(option, prior, dimred, dataset_type, C, K, piT):
    assert option == "linear"
    DTR, LTR = TRAINING_DATA()
    if dataset_type == "Z-Norm":
        from ogc.utilities import ZNormalization as znorm
        DTR = znorm(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]

    model = SVM.SVM("linear", C=C, K=K, piT=piT)
    from ogc.utilities import Kfold
    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold


def poly_svm_callback(option, prior, dimred, dataset_type, c, d, C, K):
    assert option == "polynomial"
    DTR, LTR = TRAINING_DATA()
    if dataset_type == "Z-Norm":
        from ogc.utilities import ZNormalization as znorm
        DTR = znorm(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]

    model = SVM.SVM("polynomial", c=c, d=d, C=C, K=K)
    from ogc.utilities import Kfold
    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold

def rbf_svm_callback(option, prior, dimred, dataset_type, gamma, C, K):
    assert option == "RBF"
    DTR, LTR = TRAINING_DATA()
    if dataset_type == "Z-Norm":
        from ogc.utilities import ZNormalization as znorm
        DTR = znorm(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]

    model = SVM.SVM("RBF", gamma=gamma, C=C, K=K)
    from ogc.utilities import Kfold
    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold


def main():
    # priors = [("$\pi = 0.5$", 0.5), ("$\pi = 0.1$", 0.1), ("$\pi = 0.9$", 0.9)]
    # dataset_types = [("RAW", None), ("Z-Norm", "Z-Norm")]
    # dimred = [("No PCA", None), ("PCA $(m=11)$", 11),
    #           ("PCA $(m=10)$", 10), ("PCA $(m=9)$", 9)]
    # options = [("Linear", "linear"),
    #            ("Polynomial", "polynomial"), ("RBF", "RBF")]
    # cs = [("$c = 0$", 0), ("$c = 10^{-1}", 0.1),
    #       ("$c = 10^{-2}$", 0.01), ("$c = 10^{-3}$", 0.001)]
    # ds = [("$d = 2$", 2), ("$d = 3$", 3), ("$d = 4$", 4)]
    # gammas = [("$\gamma = 1$", 1), ("$\gamma = 10$", 10),
    #           ("$\gamma = 10^2$", 100), ("$\gamma = 10^3$", 1000)]
    # Cs = [("$C = 1$", 1), ("$C = 10^{-1}$", 0.1),
    #       ("$C = 10^{-2}$", 0.01), ("$C = 10^{-3}$", 0.001)]
    # Ks = [("$K = 1$", 1), ("$K = 10^{-1}$", 0.1),
    #       ("$K = 10^{-2}$", 0.01), ("$K = 10^{-3}$", 0.001)]

    priors = [("$\pi = 0.5$", 0.5), ("$\pi = 0.1$", 0.1), ("$\pi = 0.9$", 0.9)]
    dataset_types = [("RAW", None), ("Z-Norm", "Z-Norm")]
    dimred = [("No PCA", None), ("PCA $(m=11)$", 11)]
    options = [("Linear", "linear"),
               ("Polynomial", "polynomial"), ("RBF", "RBF")]
    cs = [("$c = 0$", 0), ("$c = 10^{-2}$", 0.01)]
    ds = [("$d = 2$", 2), ("$d = 3$", 3), ("$d = 4$", 4)]
    gammas = [("$\gamma = 1$", 1),("$\gamma = 10^2$", 100)]
    Cs = [("$C = 1$", 1), ("$C = 10^{-2}$", 0.01)]
    Ks = [("$K = 1$", 1), ("$K = 10^{-2}$", 0.01)]

    _, linear_table = utilities.grid_search(
        linear_svm_callback, [options[0]], priors, dimred, dataset_types, Cs, Ks, priors)
    
    filename = TABLES_OUTPUT_PATH + "logreg_results_linear.csv"
    np.savetxt(filename, linear_table, delimiter=";",
                fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "C", "K", "\pi", "MinDCF"]))
    
    _, poly_table = utilities.grid_search(
        poly_svm_callback,[options[1]], priors, dimred, dataset_types, cs, ds, Cs, Ks)
    
    filename = TABLES_OUTPUT_PATH + "logreg_results_poly.csv"
    np.savetxt(filename, linear_table, delimiter=";",
                fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "c", "d", "C", "K", "MinDCF"]))

    _, rbf_table = utilities.grid_search(
        poly_svm_callback, [options[2]], priors, dimred, dataset_types, gammas, Cs, Ks)
    filename = TABLES_OUTPUT_PATH + "logreg_results_rbf.csv"
    np.savetxt(filename, linear_table, delimiter=";",
                fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "Gamma", "C", "K", "MinDCF"]))


    table = np.vstack([linear_table, poly_table, rbf_table])
    
    np.savetxt(TABLES_OUTPUT_PATH + "logreg_results.csv", table, delimiter=";",
               fmt="%s", header=";".join(["Prior", "Lambda", "PCA", "Dataset", "MinDCF"]))


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start} seconds")
