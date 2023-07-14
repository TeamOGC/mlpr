from typing import Dict
from os import makedirs
import matplotlib.pyplot as plt
# from ogc import dimensionality_reduction as dr
from ogc.classifiers import mvg
from ogc import utilities
import numpy.typing as npt
import numpy as np
from project import TRAINING_DATA, ROOT_PATH
import logging
from pprint import pprint
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OUTPUT_PATH = ROOT_PATH + "../images/gaussian_analysis/"
TABLES_OUTPUT_PATH = ROOT_PATH + "../tables/gaussian_analysis/"
makedirs(OUTPUT_PATH, exist_ok=True)
makedirs(TABLES_OUTPUT_PATH, exist_ok=True)

# Try all the possible combinations of MVG, TiedMVG, NaiveMVG and LDA, PCA and LDA+PCA


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

    kfold = Kfold(DTR, LTR, model, 5, prior=prior)
    return kfold


if __name__ == "__main__":
    import time
    start = time.time()
    # results = gaussian_analysis()
    fast_test = False
    if not fast_test:
        priors = [("0.5", 0.5), ("0.1", 0.1), ("0.9",0.9)]
        mvg_params = [("Standard MVG", {}), ("Naive MVG", {"naive": True}), ("Tied MVG", {"tied": True}), ("Tied Naive MVG", {"naive": True, "tied": True})]
        dataset_types = [("RAW", None), ("Z-Norm", "Z-Norm")]
        dimred = [("No PCA", None), ("PCA 5", 5)]
    else:
        priors = [("0.5", 0.5)]
        dataset_types = [("RAW", None)]
        mvg_params = [("Standard MVG", {})]
        dimred = [("No PCA", None)]

    use_csv = True
    if use_csv:
        table = utilities.load_from_csv(TABLES_OUTPUT_PATH + "mvg_results.csv")
    else:
        _, table = utilities.grid_search(mvg_callback, priors, mvg_params, dimred, dataset_types)   
        np.savetxt(TABLES_OUTPUT_PATH + "mvg_results.csv", table, delimiter=";", fmt="%s", header=";".join(["Prior", "MVG", "PCA", "Dataset", "MinDCF"]))


    print(f"Time elapsed: {time.time() - start} seconds")
