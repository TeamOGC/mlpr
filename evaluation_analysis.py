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
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/evalutaion/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)

def logreg_cb(prior):
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    DTR = dr.PCA(DTR, 5)[0]
    DTE = dr.PCA(DTE, 5)[0]
    model = LogisticRegression.LogisticRegression(l=0.1, quadratic=True, weighted=True, prior=0.5)
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)
    return metrics.minimum_detection_costs(scores, LTE, prior, 1, 1), metrics.compute_actual_DCF(scores, LTE, prior, 1, 1)
    
def mvg_cb(prior):
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    DTR = dr.PCA(DTR, 5)[0]
    DTE = dr.PCA(DTE, 5)[0]
    model = mvg.MVG(prior_probability=[prior, 1 - prior])
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)
    return metrics.minimum_detection_costs(scores, LTE, prior, 1, 1), metrics.compute_actual_DCF(scores, LTE, prior, 1, 1)

def svm_cb(prior):
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    DTR = dr.PCA(DTR, 5)[0]
    DTE = dr.PCA(DTE, 5)[0]
    model = SVM.PolynomialSVM(c=1, d=2, C=10e-1, epsilon=1)
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)
    # calibrate scores
    scores = utilities.calibrateScores(scores, LTE, scores, 0.0001)
    return metrics.minimum_detection_costs(scores, LTE, prior, 1, 1), metrics.compute_actual_DCF(scores, LTE, prior, 1, 1)

def rbf_cb(prior):
    DTR, LTR = TRAINING_DATA()
    DTE, LTE = TEST_DATA()
    DTR = dr.PCA(DTR, 5)[0]
    DTE = dr.PCA(DTE, 5)[0]
    model = SVM.RBFSVM(gamma=10e-3, C=10, K=1)
    model.fit((DTR, LTR))
    scores = model.predictAndGetScores(DTE)
    # calibrate scores
    scores = utilities.calibrateScores(scores, LTE, scores, 0.0001)
    return metrics.minimum_detection_costs(scores, LTE, prior, 1, 1), metrics.compute_actual_DCF(scores, LTE, prior, 1, 1)


def main():
    app_points = [0.1, 0.5, 0.9]
    results = []
    print("MVG")
    for pi in [i for i in app_points if True]:
        min_dcf, act_dcf = mvg_cb(pi)
        print(f"{pi=} | Minimum DCF: {min_dcf}, Actual DCF: {act_dcf}")
        results.append(("MVG", pi, min_dcf, act_dcf))
    
    print("Logistic Regression")
    for pi in [i for i in app_points if True]:
        min_dcf, act_dcf = logreg_cb(pi)
        print(f"{pi=} | Minimum DCF: {min_dcf}, Actual DCF: {act_dcf}")
        results.append(("Logistic Regression", pi, min_dcf, act_dcf))

    print("RBF SVM")
    for pi in [i for i in app_points if True]:
        min_dcf, act_dcf = rbf_cb(pi)
        print(f"{pi=} | Minimum DCF: {min_dcf}, Actual DCF: {act_dcf}")
        results.append(("RBF SVM", pi, min_dcf, act_dcf))
    

    print("Poly SVM")
    for pi in [i for i in app_points if True]:
        min_dcf, act_dcf = svm_cb(pi)
        print(f"{pi=} | Minimum DCF: {min_dcf}, Actual DCF: {act_dcf}")
        results.append(("SVM", pi, min_dcf, act_dcf))


    np.savetxt(TABLES_OUTPUT_PATH + "evaluation.csv", results, delimiter=";", fmt="%s", header="Classifier;App. Prior;Min. DCF;Act. DCF")
    

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    print(f"Time elapsed: {end-start}s")