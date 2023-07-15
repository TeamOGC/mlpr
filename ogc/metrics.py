import numpy as np
import numpy.typing as npt
from . import utilities as utils


def bayes_optimal_decisions(llr, pi1, cfn, cfp):

    threshold = -np.log(pi1*cfn/((1-pi1)*cfp))
    predictions = (llr > threshold).astype(int)
    return predictions


def detection_cost_function(M, pi1, cfn, cfp):
    FNR = M[0][1]/(M[0][1]+M[1][1])
    FPR = M[1][0]/(M[0][0]+M[1][0])

    return (pi1*cfn*FNR + (1-pi1)*cfp*FPR)


def normalized_detection_cost_function(DCF, pi1, cfn, cfp):
    dummy = np.array([pi1*cfn, (1-pi1)*cfp])
    index = np.argmin(dummy)
    return DCF/dummy[index]


def minimum_detection_costs(llr, LTE, pi1, cfn, cfp):

    sorted_llr = np.sort(llr)

    NDCF = []

    for t in sorted_llr:
        predictions = (llr > t).astype(int)

        confMatrix = utils.confusionMatrix(predictions, LTE, LTE.max()+1)
        uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)

        NDCF.append(normalized_detection_cost_function(uDCF, pi1, cfn, cfp))

    index = np.argmin(NDCF)
    return NDCF[index]


def compute_actual_DCF(llr: npt.NDArray, LTE:npt.NDArray, pi1, cfn, cfp, retRates: bool =False):

    if LTE.min() != 0:
        print("LTE must start from 0")
        LTE = LTE + 1 / 2
    t = -np.log(pi1/(1-pi1))
    predictions = (llr > t).astype(int)

    confMatrix = utils.confusionMatrix(predictions, LTE, int(LTE.max()+1))
    uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)

    NDCF = (normalized_detection_cost_function(uDCF, pi1, cfn, cfp))
    if retRates:
        # calculate fnr and fnp with confMatrix
        TPR = confMatrix[0][0]/(confMatrix[0][0]+confMatrix[0][1])
        FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
        return NDCF, TPR, FPR

    return NDCF
