from . import TRAINING_DATA, TEST_DATA, ROOT_PATH
import numpy as np
import numpy.typing as npt
from ogc.classifiers import mvg
from ogc import utilities as util
from ogc import dimensionality_reduction as dr
import matplotlib.pyplot as plt
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_PATH = ROOT_PATH + "output/gaussian_analysis/"

# Try all the possible combinations of MVG, TiedMVG, NaiveMVG and LDA, PCA and LDA+PCA

def project_PCA(m: int):
    logger.info(f"Starting PCA projection into dimension {m}")
    DTR, LTR = TRAINING_DATA()
    return dr.PCA(DTR, m)[0], LTR
    
def project_LDA(m: int):
    DTR, LTR = TRAINING_DATA()
    logger.info(f"Starting LDA projection into dimeonsion {m}")
    return dr.LDA(DTR, LTR, 1)[0], LTR

def project_PCA_LDA(m: int):
    DTR, LTR = TRAINING_DATA()
    logger.info(f"Starting PCA+LDA projection into dimension {m}")
    to_lda = dr.PCA(DTR, m)[0]
    return dr.LDA(to_lda, LTR, 1)[0], LTR

def project_MVG(training_data, test_data, mvg_params: dict):
    logger.info("Starting MVG projection")
    prior_probability = util.vcol(np.ones(2) * 1/2)
    return mvg.MVG(prior_probability=prior_probability, **mvg_params).fit(training_data).predict(test_data)

    
def gaussian_analysis():
    logger.info("Starting gaussian analysis")
    dimensionality_reductions = [(None, "No DimRed"), (project_PCA, "PCA"), (project_LDA, "LDA"), (project_PCA_LDA, "PCA+LDA")]
    classifiers = [({}, "Standard MVG"), ({"naive": True}, "Naive MVG"), ({"tied": True}, "Tied MVG"), ({"naive": True, "tied": True}, "Tied Naive MVG")]
    results: Dict[str, mvg.ClassifierResult] = {}
    for classifier, classifier_name in classifiers:
        for dimred, dimred_name in dimensionality_reductions:
            dimensions_to_test = list(range(6, 12))
            if dimred_name == "LDA":
                dimensions_to_test = [1]
            if dimred_name == "No DimRed":
                dimensions_to_test = [12]
            for dimension in dimensions_to_test:
                training_data = dimred(dimension) if dimred is not None else TRAINING_DATA()
                test_data = dimred(dimension) if dimred is not None else TEST_DATA()
                logger.info(f"Starting gaussian analysis with classifier {classifier_name} and dimensionality reduction {dimred_name}")
                result = project_MVG(training_data, test_data, classifier)
                results[f"{dimred_name};{classifier_name};{dimension}"] = result
    if logger.level == logging.DEBUG:
        print("Results:")
        print("Dimensionality Reduction;Classifier;Dimension;Accuracy")
        for res in results:
            print(f"{res};{results[res].accuracy}")

   # Plotting into a table 
    fig = plt.figure()

    axs = fig.add_subplot(111, projection='3d')
    axs.set_title("Accuracy of the different classifiers")
    dimred_axis = list()
    classif_axis = list()
    dimension_axis = list()
    c = list()
    for result, classresult in results.items():
        dimred, classifier, dimension = result.split(";")
        dimred_index = [x[1] for x in dimensionality_reductions].index(dimred)
        classifier_index = [x[1] for x in classifiers].index(classifier)
        dimension = int(dimension)
        print(dimred_index, classifier_index, dimension, classresult.accuracy )
        dimred_axis.append(dimred_index)
        classif_axis.append(classifier_index)
        dimension_axis.append(dimension)
        c.append(classresult.accuracy)
    axs.set_xlabel('# of Dimension')
    axs.set_yticks(range(len(dimensionality_reductions)), [x[1] for x in dimensionality_reductions])
    axs.set_zticks(range(len(classifiers)), [x[1] for x in classifiers])
    axs.set_xticks(range(1, 13), range(1, 13))
    img = axs.scatter(dimension_axis, dimred_axis, classif_axis, c=c, cmap='cool')
    fig.colorbar(img, orientation='vertical', label='Accuracy', pad=0.2)

    plt.savefig(OUTPUT_PATH + "gaussian_analysis.png")
    
    return results

if __name__ == "__main__":
    results = gaussian_analysis()
    