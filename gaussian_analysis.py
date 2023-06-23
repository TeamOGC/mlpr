from typing import Dict
from os import makedirs
import matplotlib.pyplot as plt
# from ogc import dimensionality_reduction as dr
from ogc.classifiers import MVG
import numpy.typing as npt
import numpy as np
from project import TRAINING_DATA, ROOT_PATH
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_PATH = ROOT_PATH + "output/gaussian_analysis/"
makedirs(OUTPUT_PATH, exist_ok=True)

# Try all the possible combinations of MVG, TiedMVG, NaiveMVG and LDA, PCA and LDA+PCA


# @cache
# def project_PCA(m: int):
#     logger.info(f"Starting PCA projection into dimension {m}")
#     DTR, LTR = TRAINING_DATA()
#     return dr.PCA(DTR, m)[0], LTR


# @cache
# def project_LDA(m: int):
#     DTR, LTR = TRAINING_DATA()
#     logger.info(f"Starting LDA projection into dimeonsion {m}")
#     return dr.LDA(DTR, LTR, 1)[0], LTR


# @cache
# def project_PCA_LDA(m: int):
#     DTR, LTR = TRAINING_DATA()
#     logger.info(f"Starting PCA+LDA projection into dimension {m}")
#     to_lda = dr.PCA(DTR, m)[0]
#     return dr.LDA(to_lda, LTR, 1)[0], LTR


# def project_MVG(training_data, test_data, mvg_params: dict):
#     logger.debug("Starting MVG projection")
#     prior_probability = util.vcol(np.ones(2) * 1/2)
#     return MVG.MVG(prior_probability=prior_probability, **mvg_params).fit(training_data).predict(test_data)


# def gaussian_analysis():
#     logger.info("Starting gaussian analysis")
#     dimensionality_reductions = [
#         (None, "No DimRed"), (project_PCA, "PCA"), (project_LDA, "LDA"), (project_PCA_LDA, "PCA+LDA")]
#     classifiers = [({}, "Standard MVG"), ({"naive": True}, "Naive MVG"), ({
#         "tied": True}, "Tied MVG"), ({"naive": True, "tied": True}, "Tied Naive MVG")]
#     results: Dict[str, mvg.ClassifierResult] = {}
#     for classifier, classifier_name in classifiers:
#         for dimred, dimred_name in dimensionality_reductions:
#             dimensions_to_test = list(range(6, 12))
#             if dimred_name == "LDA":
#                 dimensions_to_test = [1]
#             if dimred_name == "No DimRed":
#                 dimensions_to_test = [12]
#             for dimension in dimensions_to_test:
#                 training_data = dimred(
#                     dimension) if dimred is not None else TRAINING_DATA()
#                 test_data = dimred(
#                     dimension) if dimred is not None else TEST_DATA()
#                 logger.info(
#                     f"Starting gaussian analysis with classifier {classifier_name} and dimensionality reduction {dimred_name}")
#                 result = project_MVG(training_data, test_data, classifier)
#                 results[f"{dimred_name};{classifier_name};{dimension}"] = result
#     if logger.level == logging.DEBUG:
#         print("Results:")
#         print("Dimensionality Reduction;Classifier;Dimension;Accuracy")
#         for res in results:
#             print(f"{res};{results[res].accuracy}")

#    # Plotting into a table
#     fig = plt.figure()

#     axs = fig.add_subplot(111, projection='3d')
#     axs.set_title("Accuracy of the different classifiers")
#     dimred_axis = list()
#     classif_axis = list()
#     dimension_axis = list()
#     c = list()
#     for result, classresult in results.items():
#         dimred, classifier, dimension = result.split(";")
#         dimred_index = [x[1] for x in dimensionality_reductions].index(dimred)
#         classifier_index = [x[1] for x in classifiers].index(classifier)
#         dimension = int(dimension)
#         print(dimred_index, classifier_index, dimension, classresult.accuracy)
#         dimred_axis.append(dimred_index)
#         classif_axis.append(classifier_index)
#         dimension_axis.append(dimension)
#         c.append(classresult.accuracy)
#     axs.set_xlabel('# of Dimension')
#     axs.set_yticks(range(len(dimensionality_reductions)), [
#                    x[1] for x in dimensionality_reductions])
#     axs.set_zticks(range(len(classifiers)), [x[1] for x in classifiers])
#     axs.set_xticks(range(1, 13), range(1, 13))
#     img = axs.scatter(dimension_axis, dimred_axis,
#                       classif_axis, c=c, cmap='cool')
#     fig.colorbar(img, orientation='vertical', label='Accuracy', pad=0.2)

#     plt.savefig(OUTPUT_PATH + "gaussian_analysis.png")

#     return results


def MVG_RAW(prior=0.5):
    DTR, LTR = TRAINING_DATA()
    model = MVG.MVG(prior_probability=[prior, 1 - prior])
    from ogc.utilities import Kfold

    kfold = Kfold(DTR, LTR, model, 5)
    print(kfold)


if __name__ == "__main__":
    import time
    start = time.time()
    # results = gaussian_analysis()
    priors = [0.5, 0.1, 0.9]
    mvg_params = [{}, {"naive": True}, {
        "tied": True}, {"naive": True, "tied": True}]
    mvg_labels = ["Standard MVG", "Naive MVG", "Tied MVG", "Tied Naive MVG"]
    dataset_labels = ["RAW", "Z-Norm"]
    dimred = ["No DimRed", "PCA 11", "PCA 10", "PCA 9"]
    dimred_dims = [12, 11, 10, 9]
    print("Prior;Dataset;MVG;DimRed;MinDCF")
    for prior in priors:
        for i, mvg_param in enumerate(mvg_params):
            for j, dataset_label in enumerate(dataset_labels):
                for k, dimred_label in enumerate(dimred):
                    DTR, LTR = TRAINING_DATA()
                    if dataset_label == "Z-Norm":
                        from ogc.utilities import ZNormalization as znorm
                        DTR = znorm(DTR)[0]
                    if dimred_label != "No DimRed":
                        from ogc import dimensionality_reduction as dr
                        DTR = dr.PCA(DTR, dimred_dims[k])[0]
                    model = MVG.MVG(prior_probability=[
                                    prior, 1 - prior], **mvg_param)
                    from ogc.utilities import Kfold

                    mindcf = Kfold(DTR, LTR, model, 5)
                    print(
                        f"{prior};{dataset_label};{mvg_labels[i]};{dimred_label};{mindcf}")

    # MVG_RAW()
    print(f"Time elapsed: {time.time() - start} seconds")
