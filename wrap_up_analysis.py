from typing import Dict
from os import makedirs
import matplotlib.pyplot as plt
# from ogc import dimensionality_reduction as dr
from ogc.classifiers import SVM, LogisticRegression, mvg
from ogc.classifiers.gmm import GMM
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

OUTPUT_PATH = ROOT_PATH + "../images/wrap_up/"
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/wrap_up/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)


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
    kfold = Kfold(DTR, LTR, model, 5, prior=prior, act=True)
    return kfold


def mvg_callback(prior, mvg_params: dict, dimred, dataset_type):
    from ogc.utilities import Kfold
    DTR, LTR = TRAINING_DATA()
    model = mvg.MVG(prior_probability=[prior, 1 - prior], **mvg_params)
    if dataset_type == "Z-Norm":
        from ogc.utilities import ZNormalization as znorm
        DTR = znorm(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]

    kfold = Kfold(DTR, LTR, model, 5, prior=prior, act=True)
    return kfold


def logreg_callback(prior, l, dimred, dataset_type, weighted, quadratic, logreg_prior=None):
    model = LogisticRegression.LogisticRegression(
        l, logreg_prior, weighted=weighted, quadratic=quadratic)
    DTR, LTR = TRAINING_DATA()
    if dataset_type == "Z-Norm":
        from ogc.utilities import ZNormalization as znorm
        DTR = znorm(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]
    from ogc.utilities import Kfold
    kfold = Kfold(DTR, LTR, model, 5, prior=prior, act=True)
    return kfold

def GMM_callback(prior, dataset_type, mvg_param, dimred, components):
    from ogc.utilities import Kfold
    DTR, LTR = TRAINING_DATA()
    if "tied" in mvg_param.keys():
        model = GMM.GMMTiedCov(components)
    elif "naive" in mvg_param.keys():
        model = GMM.GMMDiag(components)
    else:
        model = GMM.GMM(components)

    if dataset_type == "Z-Norm":
        from ogc import utilities as utils
        DTR = utils.ZNormalization(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]

    kfold = Kfold(DTR, LTR, model, 5, prior=prior, act=True)
    return kfold


def main():
    options = [("Polynomial", "polynomial")]
    priors = [("$\pi = 0.5$", 0.5), ("$\pi = 0.1$", 0.1),
                  ("$\pi = 0.9$", 0.9)]
    dataset_types = [("RAW", None)]
    dimred = [("No PCA", None), ("PCA $(m=5)$", 5)]
    weighted = [("Weighted", True)]
    quadratic = [("Quadratic", True)]
    logreg_prior = [("$\pi_t = 0.5$", 0.5)]
    l = [("$\lambda = 0.1$", 0.1)]
    Cs = [("$C = 10^{-1}$", 0.1), ]
    Ks = [("$K = 1$", 1), ]
    cs = [("$c = 1$", 1), ]
    ds = [("$d = 2$", 2), ]
    mvg_params = [("Standard MVG", {})]
    components = [("5", 5)]
    gmm_params = [("Diagonal GMM", {"naive" : True})]
    gmm_dataset = [("Z-Norm", "Z-Norm")]


    _, final_results_poly = utilities.grid_search(poly_svm_callback,[options[0]],  priors, [dimred[1]], dataset_types, cs, ds, Cs, Ks)
    filename = TABLES_OUTPUT_PATH + "poly_svm_best.csv"
    np.savetxt(filename, final_results_poly, delimiter=";",
                    fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "c", "d", "C", "Epsilon", "actDCF"]))
    _, final_results_mvg = utilities.grid_search(mvg_callback, priors,mvg_params, dimred, dataset_types)
    filename = TABLES_OUTPUT_PATH + "mvg_best.csv"
    np.savetxt(filename, final_results_mvg, delimiter=";", fmt="%s", header=";".join(["Prior", "MVG", "PCA", "Dataset", "actDCF"]))
    _, final_results_logreg = utilities.grid_search(logreg_callback, priors, l, dimred, dataset_types, weighted, quadratic, logreg_prior)
    filename = TABLES_OUTPUT_PATH + "logreg_best.csv"
    np.savetxt(filename, final_results_logreg, delimiter=";", fmt="%s", header=";".join(["Prior", "Lambda", "PCA", "Dataset", "Weighted", "Type", "LogregPrior", "MinDCF"]))
    _, final_results_gmm = utilities.grid_search(GMM_callback, priors, gmm_dataset, gmm_params, dimred, components)
    filename = TABLES_OUTPUT_PATH + "gmm_best.csv"
    np.savetxt(filename, final_results_gmm, delimiter=";", fmt="%s", header=";".join(["Prior", "Dataset", "GMM", "PCA", "actDCF"]))

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start} seconds")