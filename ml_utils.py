# imports
import os
import numpy as np
import scipy.sparse as sparse


class TanimotoKernel:
    def __init__(self, sparse_features=True):
        self.sparse_features = sparse_features

    @staticmethod
    def similarity_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
        intersection = matrix_a.dot(matrix_b.transpose()).toarray()
        norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
        norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
        union = norm_1 + norm_2.T - intersection
        return intersection / union

    @staticmethod
    def similarity_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
        intersection = matrix_a.dot(matrix_b.transpose())
        norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
        norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
        union = np.add.outer(norm_1, norm_2.T) - intersection

        return intersection / union

    def __call__(self, matrix_a, matrix_b):
        if self.sparse_features:
            return self.similarity_from_sparse(matrix_a, matrix_b)
        else:
            raise self.similarity_from_dense(matrix_a, matrix_b)


def tanimoto_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_sparse(matrix_a, matrix_b)


def tanimoto_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_dense(matrix_a, matrix_b)


def maxminpicker(fp_list, ntopick, seed=None):
    from rdkit import SimDivFilters
    mmp = SimDivFilters.MaxMinPicker()
    n_to_pick = round(ntopick * len(fp_list))
    picks = mmp.LazyBitVectorPick(fp_list, len(fp_list), n_to_pick, seed=seed)
    return list(picks)


def untransform_data(data):
    y_predictions = []
    for y_pred in data:
        if y_pred < 5:
            y_predictions.append(5.000000)
        else:
            y_predictions.append(y_pred)
    return y_predictions


def create_directory(path: str, verbose: bool = True):
    if not os.path.exists(path):

        if len(path.split("/")) <= 2:
            os.mkdir(path)
        else:
            os.makedirs(path)
        if verbose:
            print(f"Created new directory '{path}'")
    return path
