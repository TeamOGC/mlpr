from typing import Dict
from os import makedirs
import matplotlib.pyplot as plt
# from ogc import dimensionality_reduction as dr
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

OUTPUT_PATH = ROOT_PATH + "../images/gmm_analysis/"
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/gmm_analysis/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)


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

    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold


def main():
    fast_run = True
    if fast_run:
        priors = [("$\pi = 0.5$", 0.5), ("$\pi = 0.1$", 0.1),
                  ("$\pi = 0.9$", 0.9)]
        dataset_types = [("RAW", None), ("Z-Norm", "Z-Norm")]
        mvg_params = [("Naive GMM", {
            "naive": True}), ("Tied GMM", {"tied": True})]
        dimred = [("No PCA", None), ("PCA $(m=5)$", 5)]
        components = [("5", 5)]

    else:
        priors = [("$\pi = 0.5$", 0.5), ("$\pi = 0.1$", 0.1),
                  ("$\pi = 0.9$", 0.9)]
        dataset_types = [("RAW", None), ("Z-Norm", "Z-Norm")]
        mvg_params = [("Standard GMM", {}), ("Naive GMM", {
            "naive": True}), ("Tied GMM", {"tied": True})]

        dimred = [("No PCA", None), ("PCA $(m=5)$", 5)]
        components = [("1", 1), ("2", 2), ("3", 3), ("4", 4)]

    use_csv = True 
    if use_csv:
        table = utilities.load_from_csv(TABLES_OUTPUT_PATH + "gmm_results.csv")
        table1 = utilities.load_from_csv(TABLES_OUTPUT_PATH + "gmm_results1.csv") 
    else:
        _, table = utilities.grid_search(
            GMM_callback, priors, dataset_types, mvg_params, dimred, components)

        np.savetxt(TABLES_OUTPUT_PATH + "gmm_results.csv", table, delimiter=";", fmt="%s",
                header=";".join(["Prior", "Dataset", "MVG", "PCA", "Components", "MinDCF"]))


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start} seconds")
