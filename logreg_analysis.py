from typing import Dict
from os import makedirs
import matplotlib.pyplot as plt
# from ogc import dimensionality_reduction as dr
from ogc.classifiers import LogisticRegression
from ogc import utilities
import numpy.typing as npt
import numpy as np
from project import TRAINING_DATA, ROOT_PATH
import logging
from pprint import pprint
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OUTPUT_PATH = ROOT_PATH + "../images/logreg_analysis/"
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/logreg_analysis/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)


def logreg_callback(prior, l, dimred, dataset_type, weighted):
    model = LogisticRegression.LogisticRegression(l, prior, weighted=weighted)
    DTR, LTR = TRAINING_DATA()
    if dataset_type == "Z-Norm":
        from ogc.utilities import ZNormalization as znorm
        DTR = znorm(DTR)[0]
    if dimred != None:
        from ogc import dimensionality_reduction as dr
        DTR = dr.PCA(DTR, dimred)[0]
    from ogc.utilities import Kfold
    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold


def main():
    priors = [("$\pi = 0.5$", 0.5), ("$\pi = 0.1$", 0.1), ("$\pi = 0.9$", 0.9)]
    l = [("$10^3$", 100), ("$10^2$", 10), ("$10$", 1), ("$10^{-1}$", 0.1), ("$10^{-2}$", 0.01), ("$10^{-3}$",
                                                                                                 0.001), ("$10^{-4}$", 0.0001), ("$10^{-5}$", 0.00001)]
    dataset_types = [("RAW", None), ("Z-Norm", "Z-Norm")]
    dimred = [("No PCA", None), ("PCA $(m=5)$", 5)]
    weighted = [("Weighted", True), ("Unweighted", False)]
    results, table = utilities.grid_search(
        logreg_callback, priors, l, dimred, dataset_types, weighted)
    np.savetxt(TABLES_OUTPUT_PATH + "logreg_results.csv", table, delimiter=";",
               fmt="%s", header=";".join(["Prior", "Lambda", "PCA", "Dataset", "Weighted", "MinDCF"]))


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start} seconds")
