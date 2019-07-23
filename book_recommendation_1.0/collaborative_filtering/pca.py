'''
Last update: Nov. 14 2016 by Shatian Wang

Utility Matrix: each row represents a user and each column represents a book.
PCA is used to reduce the number of rows in the utility matrix.
After PCA dimensionality reduction, each book can be represented as a lower-dimensional vector,
which makes items distances calculations less computational expensive in e.g. item-item nearest neighbor CF.
'''

import numpy as np
import uvd
import numpy.matlib


def mean_subtraction(matrix):
    """
    helper function for pca
    :param matrix: m x n matrix with each row being a measurement (user) & each col a trial (book)
    :return: the input matrix with the mean for each row subtracted off from the row
    """
    row_means = matrix.mean(1)
    return matrix - np.matlib.repmat(row_means, 1, matrix.shape[1])


def get_eigen_stuff(matrix):
    """
    helper function for pca
    :param matrix: any square matrix
    :return: eigenvalues and eigenvectors of the matrix
    with eg_vals[i] being the eigenvalue that corresponds to the eigenvector eg_vectors[,i]
    """
    cov_mat = np.cov(matrix)  # covariance matrix (checked that it's doing the correct thing)
    eg_vals, eg_vectors = numpy.linalg.eig(cov_mat)
    return eg_vals, eg_vectors


def pca(matrix):
    """
    performs pca dimensionality reduction
    :param matrix: a matrix whose size we want to reduce
    :return: the size reduced matrix (same # of cols but fewer # of rows)
    """
    mn_sbtr_spl_mat = mean_subtraction(matrix)
    eg_vals, eg_vectors = get_eigen_stuff(mn_sbtr_spl_mat)
    pca_matrix = []
    # print("sorted eigenvalues:", sorted(eg_vals, reverse = True))
    for i in range(len(eg_vals)):
        if eg_vals[i] >= 0.5: # subject to change
            pca_matrix.append(eg_vectors[:, i])
            # print("eigenvector", eg_vectors[:, i])
    pca_matrix = np.asarray(pca_matrix)
    # print("pca matrix", pca_matrix)
    return pca_matrix * matrix


def main():
    spl_mat = uvd.create_sample_matrix(8, 5)
    reduced_mat = pca(spl_mat)
    print("sample matrix:")
    print(spl_mat.toarray())
    print("#rows:")
    print(len(spl_mat.toarray()))
    print("--------------------")
    print("size reduced matrix:")
    print(reduced_mat)
    print("#rows reduced matrix:")
    print(len(reduced_mat))


if __name__ == "__main__": main()
