"""
Created on Sun Apr 23 15:19:51 2023.

@author: OGC
"""
import logging
logger = logging.getLogger(__name__)

try:
    from ..utilities import vcol, cov, vrow
    from ..density_estimation import logpdf_GAU_ND
except ImportError:
    logger.debug("Using local imports")
    from ogc.utilities import vcol, cov, vrow
    from ogc.density_estimation import logpdf_GAU_ND
    
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict
from scipy.special import logsumexp


@dataclass(init=True, repr=True)
class ClassifierResult():
    posterior_probability: npt.ArrayLike
    means_c: npt.ArrayLike
    covariances_c: npt.ArrayLike
    accuracy: float
    
    def mean(self, i: int):
        return vcol(self.means_c[:, i])

    def cov(self, i: int):
        return self.covariances_c[i, :, :]
    
    def dict(self):
        return self.__dict__
    
    def __getitem__(self, item):
       return self.__dict__[item]

class MVG:
    def __init__(self, prior_probability: npt.ArrayLike, naive: bool = False, tied: bool = False, with_log: bool = True):
        self.prior_probability = prior_probability
        self.with_log = with_log
        self.naive = naive
        self.tied = tied

    def fit(self, training_set: Tuple[npt.ArrayLike, npt.ArrayLike]):
        trn_data, trn_label = training_set
        classes = np.unique(trn_label)
        assert self.prior_probability.size == len(
            classes), f"Prior probability size ({self.prior_probability.size}) must" \
                    f" match classes count ({len(classes)})"
        # logger.debug(f"{trn_data.shape=}, {trn_label.shape=}, {classes=}, {self.prior_probability=}, {self.naive=}, {self.tied=}, {self.with_log=}")
        MU_c = np.array([trn_data[:, trn_label== i].mean(1) for i in classes]).T # mu_i = mu_c[:, i]
        COV_c = np.array([cov(trn_data[:, trn_label == i]) for i in classes]) # COV_i = COV_c[i, :, :]
        
        if self.tied:
            COV_c = COV_c.sum(axis=0) # Tied

        if self.naive:
            COV_c *= np.identity(COV_c.shape[1])
        
        self.MU_c = MU_c
        self.COV_c = COV_c

        return self
    
    def MU_i(self, i: int):
        return vcol(self.MU_c[:, i])
    
    def COV_i(self, i: int):
        if self.tied:
            return self.COV_c
        return self.COV_c[i, :, :]
    
    def predict(self, evaluation_set: Tuple[npt.NDArray, npt.NDArray]):
        ev_data, ev_label = evaluation_set
        classes = np.unique(ev_label)
        if self.with_log:
            log_score_matrix = np.array([logpdf_GAU_ND(ev_data, self.MU_i(i), self.COV_i(i)) for i in classes])
            log_SJoint = log_score_matrix + np.log(self.prior_probability)
            log_SMarginal = vrow(logsumexp(log_SJoint, axis=0))
            logSPost = log_SJoint - log_SMarginal
            SPost = np.exp(logSPost)
            predicted_labels = np.argmax(SPost, axis=0)
        else:
            score_matrix =  np.array([np.exp(logpdf_GAU_ND(ev_data, self.MU_i(i), self.COV_i(i))) for i in classes])
            SJoint = score_matrix * self.prior_probability
            SMarginal = vrow(SJoint.sum(0)) # Sums over classes
            SPost = SJoint / SMarginal
            predicted_labels = np.argmax(SPost, axis=0)
        accuracy = (predicted_labels == ev_label).sum() / ev_label.size
        logger.debug(f"{'Tied' if self.tied else ''} MVG{' (Naive)' if self.naive else ''}: {accuracy=}")
        result = ClassifierResult(SPost, self.MU_c, self.COV_c, accuracy)
        return result 



# def MVG(
#     training_set: Tuple[npt.ArrayLike, npt.ArrayLike],
#     evaluation_set: Tuple[npt.ArrayLike, npt.ArrayLike],
#     prior_probability: npt.ArrayLike,
#     with_log: bool = True,
#     naive: bool = False,
#     as_dict: bool = False
# ) -> Dict:
#     from scipy.special import logsumexp
#     trn_data, trn_label = training_set
#     ev_data, ev_label = evaluation_set

#     classes = np.unique(trn_label)
#     assert prior_probability.size == len(
#         classes), f"Prior probability size ({prior_probability.size}) must" \
#                   f" match classes count ({len(classes)})"
#     #%% Model Training
#     MU_c = np.array([trn_data[:, trn_label== i].mean(1) for i in classes]).T # mu_i = mu_c[:, i]
#     COV_c = np.array([cov(trn_data[:, trn_label == i]) for i in classes]) # COV_i = COV_c[i, :, :]
    
#     if naive:
#         COV_c *= np.identity(COV_c.shape[1])
    
#     MU_i = lambda i: vcol(MU_c[:, i])
#     COV_i = lambda i: COV_c[i, :, :]
    
#     #%% Model Evaluation
#     if with_log:
#         log_score_matrix = np.array([logpdf_GAU_ND(ev_data, MU_i(i), COV_i(i)) for i in classes])
#         log_SJoint = log_score_matrix + np.log(prior_probability)
#         log_SMarginal = vrow(logsumexp(log_SJoint, axis=0))
#         logSPost = log_SJoint - log_SMarginal
#         SPost = np.exp(logSPost)
#         predicted_labels = np.argmax(SPost, axis=0)
#     else:
#         score_matrix =  np.array([np.exp(logpdf_GAU_ND(ev_data, MU_i(i), COV_i(i))) for i in classes])
#         SJoint = score_matrix * prior_probability
#         SMarginal = vrow(SJoint.sum(0)) # Sums over classes
#         SPost = SJoint / SMarginal
#         predicted_labels = np.argmax(SPost, axis=0)
#     accuracy = (predicted_labels == ev_label).sum() / ev_label.size
#     logger.debug(f"MVG{' (Naive)' if naive else ''}: {accuracy=}")
#     result = ClassifierResult(SPost, MU_c, COV_c, accuracy)
#     return result if not as_dict else result.dict()


# def TiedMVG(
#     training_set: Tuple[npt.ArrayLike, npt.ArrayLike],
#     test_set: Tuple[npt.ArrayLike, npt.ArrayLike],
#     prior_probability: npt.ArrayLike,
#     with_log: bool = True,
#     naive: bool = False,
#     as_dict: bool = False,
#     ) -> Dict:
    
#     from scipy.special import logsumexp
#     trn_data, trn_label = training_set
#     tst_data, tst_label = test_set

#     classes = np.unique(trn_label)
#     assert prior_probability.size == len(
#         classes), f"Prior probability size ({prior_probability.size}) must" \
#                   f" match classes count ({len(classes)})"
#     #%% Model Training
#     MU_c = np.array([trn_data[:, trn_label== i].mean(1) for i in classes]).T # mu_i = mu_c[:, i]
#     COV_c = np.array([cov(trn_data[:, trn_label == i]) for i in classes]) # COV_i = COV_c[i, :, :]
#     COV_c = COV_c.sum(axis=0) # Tied
    
#     if naive:
#         COV_c *= np.identity(COV_c.shape[1])
    
#     MU_i = lambda i: vcol(MU_c[:, i])
#     COV_i = lambda i: COV_c
    
#     #%% Model Evaluation
#     if with_log:
#         log_score_matrix = np.array([logpdf_GAU_ND(tst_data, MU_i(i), COV_i(i)) for i in classes])
#         log_SJoint = log_score_matrix + np.log(prior_probability)
#         log_SMarginal = vrow(logsumexp(log_SJoint, axis=0))
#         logSPost = log_SJoint - log_SMarginal
#         SPost = np.exp(logSPost)
#         predicted_labels = np.argmax(SPost, axis=0)
#     else:
#         score_matrix =  np.array([np.exp(logpdf_GAU_ND(tst_data, MU_i(i), COV_i(i))) for i in classes])
#         SJoint = score_matrix * prior_probability
#         SMarginal = vrow(SJoint.sum(0)) # Sums over classes
#         SPost = SJoint / SMarginal
#         predicted_labels = np.argmax(SPost, axis=0)
#     accuracy = (predicted_labels == tst_label).sum() / tst_label.size
#     logger.debug(f"TiedMVG{' (Naive)' if naive else ''}: {accuracy=}")
#     result = ClassifierResult(SPost, MU_c, COV_c, accuracy)
#     return result if not as_dict else result.dict()

#%% Main
if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    