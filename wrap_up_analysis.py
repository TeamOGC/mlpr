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
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_PATH = ROOT_PATH + "../images/wrap_up/"
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/wrap_up/"
TABLES_OUTPUT_PATH_CAL = ROOT_PATH + "../tables/wrap_up/calibrated/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH_CAL, exist_ok=True)

CALIBRATE = True
CAL_LAMBDA = 0.0001 if CALIBRATE else None


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
    kfold = Kfold(DTR, LTR, model, 5, prior=prior, act=True, calibrate=CALIBRATE, lambd=CAL_LAMBDA)
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
    kfold = Kfold(DTR, LTR, model, 5, prior=prior, act=True, calibrate=CALIBRATE, lambd=CAL_LAMBDA)
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

    kfold = Kfold(DTR, LTR, model, 5, prior=prior, act=True, calibrate=CALIBRATE, lambd=CAL_LAMBDA)
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

    # chooser = [False, False, False, True]
    chooser = [True, True, False, False]

    if chooser[0]:
        mvg_filename = TABLES_OUTPUT_PATH + "mvg_best.csv"
        mvg_model_name = "Standard MVG"
        mvg_bep_filename = TABLES_OUTPUT_PATH + "mvg_bep.png"
        if CALIBRATE:
            mvg_filename = TABLES_OUTPUT_PATH_CAL + "mvg_best_calib.csv"
            mvg_model_name = "Standard MVG - Calibrated"
            mvg_bep_filename = TABLES_OUTPUT_PATH_CAL + "mvg_bep_calib.png"
        
        if use_csv:
            final_results_mvg = utilities.load_from_csv(mvg_filename)
        else:
            _, final_results_mvg = utilities.grid_search(
                mvg_callback, priors, mvg_params, [dimred[0]], dataset_types)
            np.savetxt(mvg_filename, final_results_mvg, delimiter=";", fmt="%s",
                    header=";".join(["Prior", "MVG", "PCA", "Dataset", "minDCF", "actDCF"]))
        utilities.bayesErrorPlot([float(i[-1]) for i in final_results_mvg], [float(i[-2]) for i in final_results_mvg], effPriorLogOdds, mvg_model_name, filename=mvg_bep_filename)

    if chooser[1]:
        lr_filename = TABLES_OUTPUT_PATH + "logreg_best.csv"
        lr_model_name = "Logistic Regression"
        lr_bep_filename = TABLES_OUTPUT_PATH + "logreg_bep.png"
        if CALIBRATE:
            lr_filename = TABLES_OUTPUT_PATH_CAL + "logreg_best_calib.csv"
            lr_model_name = "Logistic Regression - Calibrated"
            lr_bep_filename = TABLES_OUTPUT_PATH_CAL + "logreg_bep_calib.png"
        if use_csv:
            final_results_logreg = utilities.load_from_csv(lr_filename)
        else:
            _, final_results_logreg = utilities.grid_search(
                logreg_callback, priors, l, [dimred[1]], dataset_types, weighted, quadratic, logreg_prior)
            np.savetxt(lr_filename, final_results_logreg, delimiter=";", fmt="%s", header=";".join(
                ["Prior", "Lambda", "PCA", "Dataset", "Weighted", "Type", "LogregPrior", "MinDCF", "ActDCF"]))
        utilities.bayesErrorPlot([float(i[-1]) for i in final_results_logreg], [float(i[-2]) for i in final_results_logreg], effPriorLogOdds, lr_model_name, filename=lr_bep_filename)

    if chooser[2]:
        poly_filename = TABLES_OUTPUT_PATH + "poly_svm_best.csv"
        poly_model_name = "Polynomial SVM"
        poly_bep_filename = TABLES_OUTPUT_PATH + "poly_bep.png"
        if CALIBRATE:
            poly_filename = TABLES_OUTPUT_PATH_CAL + "poly_svm_best_calib.csv"
            poly_model_name = "Polynomial SVM - Calibrated"
            poly_bep_filename = TABLES_OUTPUT_PATH_CAL + "poly_bep_calib.png"
        if use_csv:
            final_results_poly = utilities.load_from_csv(poly_filename)
        else:
            _, final_results_poly = utilities.grid_search(poly_svm_callback, [options[0]],  priors, [
                                                        dimred[1]], dataset_types, cs, ds, Cs, Ks)
            np.savetxt(poly_filename, final_results_poly, delimiter=";",
                    fmt="%s", header=";".join(["Kernel", "Prior", "PCA", "Dataset", "c", "d", "C", "Epsilon", "actDCF"]))
        utilities.bayesErrorPlot([float(i[-1]) for i in final_results_poly], [float(i[-2]) for i in final_results_poly], effPriorLogOdds, poly_model_name, filename=poly_bep_filename)

    if chooser[3]:
        gmm_filename = TABLES_OUTPUT_PATH + "gmm_best.csv"
        gmm_model_name = "GMM"
        gmm_bep_filename = TABLES_OUTPUT_PATH + "gmm_bep.png"
        if CALIBRATE:
            gmm_filename = TABLES_OUTPUT_PATH_CAL + "gmm_best_calib.csv"
            gmm_model_name = "GMM - Calibrated"
            gmm_bep_filename = TABLES_OUTPUT_PATH_CAL + "gmm_bep_calib.png"
        if use_csv:
            final_results_gmm = utilities.load_from_csv(gmm_filename)
        else:
            _, final_results_gmm = utilities.grid_search(
                GMM_callback, priors, gmm_dataset, gmm_params, [dimred[0]], components)
            np.savetxt(gmm_filename, final_results_gmm, delimiter=";", fmt="%s",
                       header=";".join(["Prior", "Dataset", "GMM", "PCA", "actDCF"]))
        utilities.bayesErrorPlot([float(i[-1]) for i in final_results_gmm], [float(i[-2]) for i in final_results_gmm], effPriorLogOdds, gmm_model_name, filename=gmm_bep_filename)



    if(any(chooser)):
        c= "".join(['T' if i else 'F' for i in chooser])
        comparison_filename = TABLES_OUTPUT_PATH + f"comparison-{c}.png"
        if CALIBRATE:
            comparison_filename = TABLES_OUTPUT_PATH_CAL + f"comparison_calib-{c}.png"
        comparison = []
        if chooser[0]:
            comparison.append(([float(i[-2]) for i in final_results_mvg], [float(i[-1]) for i in final_results_mvg], mvg_model_name))
        if chooser[1]:
            comparison.append(([float(i[-2]) for i in final_results_logreg], [float(i[-1]) for i in final_results_logreg], lr_model_name))
        if chooser[2]:
            comparison.append(([float(i[-2]) for i in final_results_poly], [float(i[-1]) for i in final_results_poly], poly_model_name))
        if chooser[3]:
            comparison.append(([float(i[-2]) for i in final_results_gmm], [float(i[-1]) for i in final_results_gmm], gmm_model_name))
        utilities.multiple_bep(effPriorLogOdds, comparison, filename=comparison_filename)

    # return (final_results_mvg, final_results_logreg, final_results_poly, final_results_gmm)

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    

    print(f"Time elapsed: {time.time() - start} seconds")
