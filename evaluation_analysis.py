from typing import Dict
from os import makedirs
import matplotlib.pyplot as plt
from ogc import dimensionality_reduction as dr
from ogc.classifiers import SVM, LogisticRegression, mvg
from ogc.classifiers.gmm import GMM
from ogc import utilities, metrics
import numpy.typing as npt
import numpy as np
from project import TRAINING_DATA, ROOT_PATH, TEST_DATA
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_PATH = ROOT_PATH + "../images/evaluation/"
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/evaluation/"
TABLES_OUTPUT_PATH_CAL = ROOT_PATH + "../tables/evaluation/calibrated/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH_CAL, exist_ok=True)

CALIBRATE = True
CAL_LAMBDA = 0.0001 if CALIBRATE else None


def poly_svm_callback(prior):
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    DTR = dr.PCA(DTR, 5)[0]
    DTE = dr.PCA(DTE, 5)[0]
    model = SVM.PolynomialSVM(c=1, d=2, C=10e-1, epsilon=1)
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)

    mindcf = metrics.minimum_detection_costs(scores, LTE, prior, 1, 1)
    actdcf, TPR, fpr = metrics.compute_actual_DCF(
        scores, LTE, prior, 1, 1, retRates=True)
    # Calibrate
    calibratedScores = utilities.calibrateScores(scores, LTE, scores, 0.0001)
    calmindcf = metrics.minimum_detection_costs(
        calibratedScores, LTE, prior, 1, 1)
    calactdcf, cal_TPR, cal_fpr = metrics.compute_actual_DCF(
        calibratedScores, LTE, prior, 1, 1, retRates=True)

    return (mindcf, actdcf, calmindcf, calactdcf, TPR, fpr, cal_TPR, cal_fpr)


def mvg_callback(prior):
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    model = mvg.MVG(prior_probability=[prior, 1 - prior])
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)

    mindcf = metrics.minimum_detection_costs(scores, LTE, prior, 1, 1)
    actdcf, TPR, fpr = metrics.compute_actual_DCF(
        scores, LTE, prior, 1, 1, retRates=True)
    # Calibrate
    calibratedScores = utilities.calibrateScores(scores, LTE, scores, 0.0001)
    calmindcf = metrics.minimum_detection_costs(
        calibratedScores, LTE, prior, 1, 1)
    calactdcf, cal_TPR, cal_fpr = metrics.compute_actual_DCF(
        calibratedScores, LTE, prior, 1, 1, retRates=True)

    return (mindcf, actdcf, calmindcf, calactdcf, TPR, fpr, cal_TPR, cal_fpr)


def logreg_callback(prior):
    model = LogisticRegression.LogisticRegression(
        0.1, 0.5, weighted=True, quadratic=True)
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    DTR = dr.PCA(DTR, 5)[0]
    DTE = dr.PCA(DTE, 5)[0]
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)

    mindcf = metrics.minimum_detection_costs(scores, LTE, prior, 1, 1)
    actdcf, TPR, fpr = metrics.compute_actual_DCF(
        scores, LTE, prior, 1, 1, retRates=True)
    # Calibrate
    calibratedScores = utilities.calibrateScores(scores, LTE, scores, 0.0001)
    calmindcf = metrics.minimum_detection_costs(
        calibratedScores, LTE, prior, 1, 1)
    calactdcf, cal_TPR, cal_fpr = metrics.compute_actual_DCF(
        calibratedScores, LTE, prior, 1, 1, retRates=True)

    return (mindcf, actdcf, calmindcf, calactdcf, TPR, fpr, cal_TPR, cal_fpr)


def GMM_callback(prior):
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    components = 5
    model = GMM.GMMDiag(components)
    DTR = utilities.ZNormalization(DTR)[0]
    DTE = utilities.ZNormalization(DTE)[0]
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)

    mindcf = metrics.minimum_detection_costs(scores, LTE, prior, 1, 1)
    actdcf, TPR, fpr = metrics.compute_actual_DCF(
        scores, LTE, prior, 1, 1, retRates=True)
    # Calibrate
    calibratedScores = utilities.calibrateScores(scores, LTE, scores, 0.0001)
    calmindcf = metrics.minimum_detection_costs(
        calibratedScores, LTE, prior, 1, 1)
    calactdcf, cal_TPR, cal_fpr = metrics.compute_actual_DCF(
        calibratedScores, LTE, prior, 1, 1, retRates=True)

    return (mindcf, actdcf, calmindcf, calactdcf, TPR, fpr, cal_TPR, cal_fpr)


def rbf_svm_callback(prior):
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    DTR = dr.PCA(DTR, 5)[0]
    DTE = dr.PCA(DTE, 5)[0]
    model = SVM.RBFSVM(gamma=10e-3, C=10, K=1)
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)

    mindcf = metrics.minimum_detection_costs(scores, LTE, prior, 1, 1)
    actdcf, TPR, fpr = metrics.compute_actual_DCF(
        scores, LTE, prior, 1, 1, retRates=True)
    # Calibrate
    calibratedScores = utilities.calibrateScores(scores, LTE, scores, 0.0001)
    calmindcf = metrics.minimum_detection_costs(
        calibratedScores, LTE, prior, 1, 1)
    calactdcf, cal_TPR, cal_fpr = metrics.compute_actual_DCF(
        calibratedScores, LTE, prior, 1, 1, retRates=True)

    return (mindcf, actdcf, calmindcf, calactdcf, TPR, fpr, cal_TPR, cal_fpr)


def main():
    numberOfPoints = 18
    effPriorLogOdds = np.linspace(-3, 3, numberOfPoints)
    effPriors = 1/(1+np.exp(-1*effPriorLogOdds))
    # effPriors = [0.1, 0.5, 0.9]
    # effPriorLogOdds = [np.log(p/(1-p)) for p in effPriors]
    priors = [(f"$\pi_T = {p:.3f}$", p) for p in effPriors]

    use_csv: bool = False

    chooser = [True, True, True, True, True]
    # chooser = [True, True, False, False, True]

    if chooser[0]:
        mvg_filename = TABLES_OUTPUT_PATH + "mvg_best.csv"
        mvg_model_name = "Standard MVG ?"
        mvg_bep_filename = TABLES_OUTPUT_PATH + "mvg_bep ?.png"

        if use_csv:
            final_results_mvg = utilities.load_from_csv(mvg_filename)
        else:
            _, final_results_mvg = utilities.grid_search(mvg_callback, priors)
            np.savetxt(mvg_filename, final_results_mvg, delimiter=";", fmt="%s",
                       header=";".join(["Prior", "minDCF", "actDCF", "calMinDCF", "calActDCF", "TPR", "FPR", "calTPR", "calFPR"]))
        utilities.bayesErrorPlot([float(i[2]) for i in final_results_mvg], [float(i[1])
                                 for i in final_results_mvg], effPriorLogOdds, mvg_model_name.replace("?", ""), filename=mvg_bep_filename.replace("?", ""))
        utilities.bayesErrorPlot([float(i[4]) for i in final_results_mvg], [float(i[3])
                                                                            for i in final_results_mvg], effPriorLogOdds, mvg_model_name.replace("?", "- Calibrated"), filename=mvg_bep_filename.replace("?", "calib"))

    if chooser[1]:
        lr_filename = TABLES_OUTPUT_PATH + "logreg_best.csv"
        lr_model_name = "Logistic Regression ?"
        lr_bep_filename = TABLES_OUTPUT_PATH + "logreg_bep ?.png"
        if use_csv:
            final_results_logreg = utilities.load_from_csv(lr_filename)
        else:
            _, final_results_logreg = utilities.grid_search(
                logreg_callback, priors)
            np.savetxt(lr_filename, final_results_logreg, delimiter=";", fmt="%s", header=";".join(
                ["Prior", "MinDCF", "ActDCF", "CalMinDCF", "CalActDCF", "TPR", "FPR", "CalTPR", "CalFPR"]))
        utilities.bayesErrorPlot([float(i[2]) for i in final_results_logreg], [float(i[1])
                                 for i in final_results_logreg], effPriorLogOdds, lr_model_name.replace("?", ""), filename=lr_bep_filename.replace("?", ""))
        utilities.bayesErrorPlot([float(i[4]) for i in final_results_logreg], [float(i[3])
                                                                               for i in final_results_logreg], effPriorLogOdds, lr_model_name.replace("?", "- Calibrated"), filename=lr_bep_filename.replace("?", "calib"))

    if chooser[2]:
        poly_filename = TABLES_OUTPUT_PATH + "poly_svm.csv"
        poly_model_name = "Polynomial SVM ?"
        poly_bep_filename = TABLES_OUTPUT_PATH + "poly_bep ?.png"

        if use_csv:
            final_results_poly = utilities.load_from_csv(poly_filename)
        else:
            _, final_results_poly = utilities.grid_search(
                poly_svm_callback, priors)
            np.savetxt(poly_filename, final_results_poly, delimiter=";",
                       fmt="%s", header=";".join(["Prior", "minDCF", "actDCF", "calMinDCF", "calActDCF", "TPR", "FPR", "calTPR", "calFPR"]))
        utilities.bayesErrorPlot([float(i[2]) for i in final_results_poly], [float(i[1])
                                 for i in final_results_poly], effPriorLogOdds, poly_model_name.replace("?", ""), filename=poly_bep_filename.replace("?", ""))
        utilities.bayesErrorPlot([float(i[4]) for i in final_results_poly], [float(i[3])
                                                                             for i in final_results_poly], effPriorLogOdds, poly_model_name.replace("?", "- Calibrated"), filename=poly_bep_filename.replace("?", "calib"))

    if chooser[3]:
        gmm_filename = TABLES_OUTPUT_PATH + "gmm_best.csv"
        gmm_model_name = "GMM ?"
        gmm_bep_filename = TABLES_OUTPUT_PATH + "gmm_bep ?.png"
        if use_csv:
            final_results_gmm = utilities.load_from_csv(gmm_filename)
        else:
            _, final_results_gmm = utilities.grid_search(GMM_callback, priors)
            np.savetxt(gmm_filename, final_results_gmm, delimiter=";", fmt="%s",
                       header=";".join(["Prior", "minDCF", "actDCF", "calMinDCF", "calActDCF", "TPR", "FPR", "calTPR", "calFPR"]))
        utilities.bayesErrorPlot([float(i[2]) for i in final_results_gmm], [float(i[1])
                                 for i in final_results_gmm], effPriorLogOdds, gmm_model_name.replace("?", ""), filename=gmm_bep_filename.replace("?", ""))
        utilities.bayesErrorPlot([float(i[4]) for i in final_results_gmm], [float(i[3])
                                                                            for i in final_results_gmm], effPriorLogOdds, gmm_model_name.replace("?", "- Calibrated"), filename=gmm_bep_filename.replace("?", "calib"))
    if chooser[4]:
        # rbf
        rbf_filename = TABLES_OUTPUT_PATH + "rbf_svm.csv"
        rbf_model_name = "RBF SVM ?"
        rbf_bep_filename = TABLES_OUTPUT_PATH + "rbf_bep ?.png"
        if use_csv:
            final_results_rbf = utilities.load_from_csv(rbf_filename)
        else:
            _, final_results_rbf = utilities.grid_search(
                rbf_svm_callback, priors)
            np.savetxt(rbf_filename, final_results_rbf, delimiter=";", fmt="%s",
                       header=";".join(["Prior", "minDCF", "actDCF", "calMinDCF", "calActDCF", "TPR", "FPR", "calTPR", "calFPR"]))
        utilities.bayesErrorPlot([float(i[2]) for i in final_results_rbf], [float(i[1])
                                 for i in final_results_rbf], effPriorLogOdds, rbf_model_name.replace("?", ""), filename=rbf_bep_filename.replace("?", ""))
        utilities.bayesErrorPlot([float(i[4]) for i in final_results_rbf], [float(i[3])
                                                                            for i in final_results_rbf], effPriorLogOdds, rbf_model_name.replace("?", "- Calibrated"), filename=rbf_bep_filename.replace("?", "calib"))

    if (len([c for c in chooser if c]) > 1):
        c = "".join(['T' if i else 'F' for i in chooser])
        comparison_filename = TABLES_OUTPUT_PATH + f"comparison_?_-{c}.png"
        comparison = []
        comparison_calib = []
        fpr_list = []
        fpr_list_calib = []
        if chooser[0]:
            comparison.append(([float(i[2]) for i in final_results_mvg], [float(
                i[1]) for i in final_results_mvg], mvg_model_name.replace("?", "")))
            comparison_calib.append(([float(i[4]) for i in final_results_mvg],
                                     [float(i[3]) for i in final_results_mvg], mvg_model_name.replace("?", "- Calibrated")))
            fpr_list.append(([float(i[5]) for i in final_results_mvg], [float(
                i[6]) for i in final_results_mvg], mvg_model_name.replace("?", "")))
            fpr_list_calib.append(([float(i[7]) for i in final_results_mvg], [float(
                i[8]) for i in final_results_mvg], mvg_model_name.replace("?", "- Calibrated")))

        if chooser[1]:
            comparison.append(([float(i[2]) for i in final_results_logreg], [
                              float(i[1]) for i in final_results_logreg], lr_model_name))
            comparison_calib.append(([float(i[4]) for i in final_results_logreg],
                                     [float(i[3]) for i in final_results_logreg], lr_model_name.replace("?", "- Calibrated")))
            fpr_list.append(([float(i[5]) for i in final_results_logreg], [float(
                i[6]) for i in final_results_logreg], lr_model_name.replace("?", "")))
            fpr_list_calib.append(([float(i[7]) for i in final_results_logreg], [float(
                i[8]) for i in final_results_logreg], lr_model_name.replace("?", "- Calibrated")))
        if chooser[2]:
            comparison.append(([float(i[2]) for i in final_results_poly], [
                              float(i[1]) for i in final_results_poly], poly_model_name))
            comparison_calib.append(([float(i[4]) for i in final_results_poly],
                                     [float(i[3]) for i in final_results_poly], poly_model_name.replace("?", "- Calibrated")))
            fpr_list.append(([float(i[5]) for i in final_results_poly], [float(
                i[6]) for i in final_results_poly], poly_model_name.replace("?", "")))
            fpr_list_calib.append(([float(i[7]) for i in final_results_poly], [float(
                i[8]) for i in final_results_poly], poly_model_name.replace("?", "- Calibrated")))
        if chooser[3]:
            comparison.append(([float(i[2]) for i in final_results_gmm], [
                              float(i[1]) for i in final_results_gmm], gmm_model_name))
            comparison_calib.append(([float(i[4]) for i in final_results_gmm],
                                     [float(i[3]) for i in final_results_gmm], gmm_model_name.replace("?", "- Calibrated")))
            fpr_list.append(([float(i[5]) for i in final_results_gmm], [float(
                i[6]) for i in final_results_gmm], gmm_model_name.replace("?", "")))
            fpr_list_calib.append(([float(i[7]) for i in final_results_gmm], [float(
                i[8]) for i in final_results_gmm], gmm_model_name.replace("?", "- Calibrated")))
        if chooser[4]:
            comparison.append(([float(i[2]) for i in final_results_rbf], [
                              float(i[1]) for i in final_results_rbf], rbf_model_name))
            comparison_calib.append(([float(i[4]) for i in final_results_rbf],
                                     [float(i[3]) for i in final_results_rbf], rbf_model_name.replace("?", "- Calibrated")))
            fpr_list.append(([float(i[5]) for i in final_results_rbf], [float(
                i[6]) for i in final_results_rbf], rbf_model_name.replace("?", "")))
            fpr_list_calib.append(([float(i[7]) for i in final_results_rbf], [float(
                i[8]) for i in final_results_rbf], rbf_model_name.replace("?", "- Calibrated")))
        utilities.multiple_bep(effPriorLogOdds, comparison,
                               filename=comparison_filename.replace("?", ""))
        utilities.multiple_bep(effPriorLogOdds, comparison_calib,
                               filename=comparison_filename.replace("?", "calib"))
        utilities.plot_roc(
            fpr_list, filename=comparison_filename.replace("?", "det"))
        utilities.plot_roc(
            fpr_list_calib, filename=comparison_filename.replace("?", "det_calib"))

    # return (final_results_mvg, final_results_logreg, final_results_poly, final_results_gmm)


if __name__ == "__main__":
    import time
    print("---- Calibration: ", CALIBRATE)
    start = time.time()
    main()

    print(f"Time elapsed: {time.time() - start} seconds")
