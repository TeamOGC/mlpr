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
logger.setLevel(logging.INFO)

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
    kfold = Kfold(DTR, LTR, model, 5, prior=prior, act=True, calibrate=True, lambd=0.01)
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
    numberOfPoints=18
    effPriorLogOdds = np.linspace(-3, 3, numberOfPoints)
    effPriors = 1/(1+np.exp(-1*effPriorLogOdds))

    options = [("Polynomial", "polynomial")]
    priors = [(f"$\pi_T = {p:.3f}$", p) for p in effPriors]
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
    gmm_params = [("Diagonal GMM", {"naive": True})]
    gmm_dataset = [("Z-Norm", "Z-Norm")]

    use_csv: bool = False

    # chooser = [False, False, True, True]
    chooser = [False, True, False, False]

    if chooser[0]:
        filename = TABLES_OUTPUT_PATH + "mvg_best.csv"
        if use_csv:
            final_results_mvg = utilities.load_from_csv(filename)
        else:
            _, final_results_mvg = utilities.grid_search(
                mvg_callback, priors, mvg_params, [dimred[0]], dataset_types)
            np.savetxt(filename, final_results_mvg, delimiter=";", fmt="%s",
                    header=";".join(["Prior", "MVG", "PCA", "Dataset", "minDCF", "actDCF"]))
        utilities.bayesErrorPlot([float(i[-1]) for i in final_results_mvg], [float(i[-2]) for i in final_results_mvg], effPriorLogOdds, "Standard MVG", filename=TABLES_OUTPUT_PATH + "mvg_bayes_error.png")

    if chooser[1]:
        filename = TABLES_OUTPUT_PATH + "logreg_best_calib.csv"
        if use_csv:
            final_results_logreg = utilities.load_from_csv(filename)
        else:
            _, final_results_logreg = utilities.grid_search(
                logreg_callback, priors, l, [dimred[1]], dataset_types, weighted, quadratic, logreg_prior)
            np.savetxt(filename, final_results_logreg, delimiter=";", fmt="%s", header=";".join(
                ["Prior", "Lambda", "PCA", "Dataset", "Weighted", "Type", "LogregPrior", "MinDCF", "ActDCF"]))
        utilities.bayesErrorPlot([float(i[-1]) for i in final_results_logreg], [float(i[-2]) for i in final_results_logreg], effPriorLogOdds, "Logistic Regression - Calibrated", filename=TABLES_OUTPUT_PATH + "logreg_bayes_error_calibrated.png")

    if chooser[2]:
        filename = TABLES_OUTPUT_PATH + "poly_svm_best.csv"
        if use_csv:
            final_results_poly = utilities.load_from_csv(filename)
        else:
            _, final_results_poly = utilities.grid_search(poly_svm_callback, [options[0]],  priors, [
                                                        dimred[1]], dataset_types, cs, ds, Cs, Ks)
            np.savetxt(filename, final_results_poly, delimiter=";",
                    fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "c", "d", "C", "Epsilon", "actDCF"]))
        utilities.bayesErrorPlot([float(i[-1]) for i in final_results_poly], [float(i[-2]) for i in final_results_poly], effPriorLogOdds, "Polynomial SVM", filename=TABLES_OUTPUT_PATH + "poly_bayes_error.png")

    if chooser[3]:
        filename = TABLES_OUTPUT_PATH + "gmm_best.csv"
        if use_csv:
            final_results_gmm = utilities.load_from_csv(filename)
        else:
            _, final_results_gmm = utilities.grid_search(
                GMM_callback, priors, gmm_dataset, gmm_params, [dimred[0]], components)
            np.savetxt(filename, final_results_gmm, delimiter=";", fmt="%s",
                       header=";".join(["Prior", "Dataset", "GMM", "PCA", "actDCF"]))
        utilities.bayesErrorPlot([float(i[-1]) for i in final_results_gmm], [float(i[-2]) for i in final_results_gmm], effPriorLogOdds, "GMM", filename=TABLES_OUTPUT_PATH + "gmm_bayes_error.png")





    # return (final_results_mvg, final_results_logreg, final_results_poly, final_results_gmm)

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    

    print(f"Time elapsed: {time.time() - start} seconds")
