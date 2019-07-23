'''
Jinny, JordiKai, and Shatian
'''

import numpy as np
import scipy.sparse as sp
import math
import random
import cProfile
import re
from scipy.sparse import csr_matrix
from random import shuffle


def load_sparse_csr(filename):
        '''
        http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
        '''
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def create_sample_matrix(n, m, p):
    '''
    :param n: number of rows in the matrix subset
    :param m: number of columns in the matrix subset
    :return: a subset of the matrix for testing purposes
    '''
    # originalMatrix = load_sparse_csr("utilityMat.npz")
    zeros_in_ratings_list = (math.ceil((9*p)/(1- p)))*[0]
    non_zeros_ratings_list = list(range(1, 10))
    row_list = []
    for i in range(n):
        row = []
        for j in range(m):
            # rand_num = randint(0, 9)
            rand_num = random.choice(zeros_in_ratings_list + non_zeros_ratings_list)
            row.append(rand_num)
        row_list.append(row)
    return csr_matrix(row_list)


def pre_process(matrix):
    ''' STEP 1
    :param matrix: utility matrix
    :return the normalized matrix
    http://stackoverflow.com/questions/39685168/scipy-sparse-matrix-subtract-row-mean-to-nonzero-elements
    '''
    # find the average of the rows
    (r,c,d)=sp.find(matrix)
    countings=np.bincount(r)
    sums=np.bincount(r,weights=d)
    averages = []
    for i in range(len(countings)):
        if countings[i] == 0:
            averages.append(0)
        else:
            averages.append(sums[i]/countings[i])
    # creates a matrix with the averages as the diagonal
    diag_averages = sp.diags(averages, 0)
    matrix_copy = matrix.copy()
    # creates a matrix of zero and ones preserving all non-values
    matrix_copy.data = np.ones_like(matrix_copy.data)
    # returns matrix where all non-zero values have been subtracted from mean
    return (matrix - diag_averages*matrix_copy), averages


def get_matrix_average(matrix):
    ''' Helper function that calculates the average of a matrix, does NOT include zeros
    :param matrix
    :return: the average of all non-zero values in matrix
    '''
    sums = matrix.sum(axis=1).A1
    counts = np.diff(matrix.indptr)
    return(sum(sums) / sum(counts))


def init_uv(d, matrix):
    ''' STEP 2
    :param d: setting d for calculating U & V
    :param matrix: original matrix that we extrapolate row and col count from
    :return: U & V matrix
    '''
    num_row = matrix.shape[0]
    num_col = matrix.shape[1]
    u = np.ndarray(shape=(num_row,d), dtype=float)
    avg = get_matrix_average(matrix)
    # divide by float(d) to make sure that it is not doing integer division
    u.fill(math.sqrt(avg/float(d)))
    v = np.ndarray(shape=(d,num_col), dtype=float)
    v.fill(math.sqrt(avg/float(d)))
    # print(math.sqrt(avg/float(d)))
    return u, v


def init_uv_all_one(d, matrix):
    ''' STEP 2
    :param d: setting d for calculating U & V
    :param matrix: original matrix that we extrapolate row and col count from
    :return: U & V matrix
    '''
    num_row = matrix.shape[0]
    num_col = matrix.shape[1]
    u = np.ndarray(shape=(num_row,d), dtype=float)
    u.fill(1)
    v = np.ndarray(shape=(d,num_col), dtype=float)
    v.fill(1)
    return u, v


def update_u(r, s, u, v, normalized_matrix, nonzero_indices):
    ''' Helper function for optimize that calculates the value to insert at position in matrix U
    CH 9 Recommendation Systems (p332-335)
    :param r: row num of cell updating
    :param s: col num of cell updating
    :param u: matrix generated in step 2, dimensions n x d
    :param v: matrix generated in step 2, dimensions d x m
    :param normalized_matrix: normalized matrix, dimensions n x m
    :return: Value to insert at u[r, s]
    '''
    # calculate numerator
    numerator = 0
    for j in range(normalized_matrix.shape[1]):
        # print("Type of nonzero_indicies", type(nonzero_indices))
        # print([r, j] not in nonzero_indices)
        if (r, j) not in nonzero_indices:
            continue
        inner_sum = 0
        for k in range(u.shape[1]):
            if k != s:
                inner_sum += u[r, k]*v[k, j]
        # print("inner sum ", inner_sum)
        # print("m[0, 1] ", normalized_matrix[r, j])
        # print(normalized_matrix[r, j] - inner_sum)
        # print("v[s, j]", v[s, j])
        numerator += v[s, j]*(normalized_matrix[r, j] - inner_sum)
        # print("numerator", numerator)
    # calculate denominator
    denominator = 0
    for j in range(v.shape[1]):
        if (r,j) in nonzero_indices:
            denominator += v[s,j]**2
    new_value = numerator/denominator
    # print("value", new_value)
    u[r, s] = new_value
    # print(u[r, s])


def update_v(r, s, u, v, normalized_matrix, nonzero_indices):
    ''' Helper function for optimize that calculates the value to insert at position in matrix V
    CH 9 Recommendation Systems (p332-335)
    :param r: row num of cell updating
    :param s: col num of cell updating
    :param u: matrix generated in step 2, dimensions n x d
    :param v: matrix generated in step 2, dimensions d x m
    :param normalized_matrix: original matrix, dimensions n x m
    :return: Value to insert at v[r, s]
    '''
    # numerator
    numerator = 0
    for i in range(normalized_matrix.shape[0]):
        if (i, s) not in nonzero_indices:
            continue
        inner_sum = 0
        for k in range(u.shape[1]):
            if k != r:
                inner_sum += u[i, k]*v[k, s]
        numerator += u[i, r]*(normalized_matrix[i, s] - inner_sum)
    # print("numerator", numerator)
    # denominator
    denominator = 0
    for i in range(u.shape[0]):
        if (i,s) in nonzero_indices:
            denominator += u[i,r]**2
    # print("denominator", denominator)
    new_value = numerator/denominator
    # print("value", new_value)
    v[r, s] = new_value
    # print(v[r, s])


def track_rated(og_matrix):
    # grab the coordinates of all nonzero elements
    # Return hashset of coord (tuples)
    nonzero_ndarray = np.transpose(np.nonzero(og_matrix))
    nonzero_hashset = set()
    for coord in nonzero_ndarray:
        nonzero_hashset.add(tuple(coord))
    return nonzero_hashset


def rmse(u, v, normalized_matrix, nonzero_indices):
    # (P329) Sum over all nonblank entries in normalized_matrix
    # The square difference btwn that entry and the corresponding entry in UV product
    uv_product = np.matmul(u, v)
    squared_sum = 0
    for (i, j) in nonzero_indices:
        squared_sum += (normalized_matrix[i,j]-uv_product[i,j])**2
    average_squared_sum = squared_sum / len(nonzero_indices)
    rmse = math.sqrt(average_squared_sum)
    return rmse


def optimize_by_row(u, v, normalized_matrix, nonzero_indices):
    # We will first try to update row by row, let r,s be a cell in U or V
    threshold = 0.005
    prev_rmse = rmse(u, v, normalized_matrix, nonzero_indices)
    done = False
    measure_threshold = []
    # stop gradient descent when decrease in rmse is smaller than the threshold
    while not done:
        # update u
        for r in range(u.shape[0]):
            for s in range(u.shape[1]):
                update_u(r, s, u, v, normalized_matrix, nonzero_indices)
                # post_rmse = rmse(u, v, normalized_matrix, nonzero_indices) #remove
                # print("U: ", post_rmse) #remove

        # update v
        for r in range(v.shape[0]):
            for s in range(v.shape[1]):
                update_v(r, s, u, v, normalized_matrix, nonzero_indices)
                # post_rmse = rmse(u, v, normalized_matrix, nonzero_indices) #remove
                # print("V: ", post_rmse) #remove

        post_rmse = rmse(u, v, normalized_matrix, nonzero_indices)
        decrease = prev_rmse - post_rmse
        prev_rmse = post_rmse
        # print("    Decrease Value ", decrease)
        if decrease < threshold:
            done = True
        # print(u, "\n", v)


def optimize_by_col(u, v, normalized_matrix, nonzero_indices):
    # We will first try to update row by row, let r,s be a cell in U or V
    threshold = 0.005
    prev_rmse = rmse(u, v, normalized_matrix, nonzero_indices)
    done = False
    # stop gradient descent when decrease in rmse is smaller than the threshold
    while not done:
        # update u
        for s in range(u.shape[1]):
            for r in range(u.shape[0]):
                update_u(r, s, u, v, normalized_matrix, nonzero_indices)
                # post_rmse = rmse(u, v, normalized_matrix, nonzero_indices) #remove
                # print("U: ", post_rmse) #remove

        # update v
        for s in range(v.shape[1]):
            for r in range(v.shape[0]):
                update_v(r, s, u, v, normalized_matrix, nonzero_indices)
                # post_rmse = rmse(u, v, normalized_matrix, nonzero_indices) #remove
                # print("V: ", post_rmse) #remove

        post_rmse = rmse(u, v, normalized_matrix, nonzero_indices)
        decrease = prev_rmse - post_rmse
        prev_rmse = post_rmse
        # print("    Decrease Value ", decrease)
        if decrease < threshold:
            done = True


def create_uv_index_lists(u, v):
    u_index_lst = []
    v_index_lst = []
    for s in range(u.shape[1]):
        for r in range(u.shape[0]):
            u_index_lst.append((r, s))
    for s in range(v.shape[1]):
        for r in range(v.shape[0]):
            v_index_lst.append((r, s))
    return u_index_lst, v_index_lst


def optimize_random(u, v, normalized_matrix, nonzero_indices):
    # We will first try to update row by row, let r,s be a cell in U or V
    u_index_lst, v_index_lst = create_uv_index_lists(u, v)
    # stop gradient descent when decrease in rmse is smaller than the threshold
    threshold = 0.005
    prev_rmse = rmse(u, v, normalized_matrix, nonzero_indices)
    done = False
    while not done:
        shuffle(u_index_lst)
        shuffle(v_index_lst)
        for coord in u_index_lst:
            r, s = coord
            update_u(r, s, u, v, normalized_matrix, nonzero_indices)
            # u_rmse = rmse(u, v, normalized_matrix, nonzero_indices) #remove
            # print("U: ", u_rmse) #remove
        for coord in v_index_lst:
            r, s = coord
            update_v(r, s, u, v, normalized_matrix, nonzero_indices)
            # v_rmse = rmse(u, v, normalized_matrix, nonzero_indices) #remove
            # print("V: ", v_rmse) #remove
        post_rmse = rmse(u, v, normalized_matrix, nonzero_indices)
        # print(counter, ":    Prev RMSE ", prev_rmse, "    Post RMSE ", post_rmse)
        decrease = prev_rmse - post_rmse
        prev_rmse = post_rmse
        # print("    Decrease Value ", decrease)
        if decrease < threshold:
            done = True


def main():
    """
    PROFILING
    To create profile of uvd.py run the following cmd:
        python3 -m cProfile -o profile_uvd.txt uvd.py
    This saves the profile to a file. This file then can be read and sorted by using
    the read_profiles.py file, given the txt file as an arg in the cmd ln
    """
    sample_matrix = create_sample_matrix(100, 150, 0.9)
    # Size of dataset n = 35,000,  m = 600,000, p = 0.99
    nonzero_indices = track_rated(sample_matrix)
    # print("Test Matrix \n", sample_matrix.toarray())

    # Step 1 Pre-process matrix
    sample_matrix_normalized, avg_each_row = pre_process(sample_matrix)
    sample_matrix_t = sample_matrix_normalized.transpose(copy=False)
    sample_matrix_normalized, avg_each_col = pre_process(sample_matrix_t)
    sample_matrix_normalized = sample_matrix_normalized.transpose(copy=False)
    # print("Normalized Matrix \n", sample_matrix_normalized.toarray())

    # Step 2 Generate U and V based on d and original matrix
    u, v = init_uv(3, sample_matrix)

    # Step 3 Optimization
    min_rmse = rmse(u, v, sample_matrix_normalized, nonzero_indices)
    best_u = u
    best_v = v

    # Random Optimization
    print("----- Optimize random -----")
    for i in range(4):
        u_rand = u.copy()
        v_rand = v.copy()
        normalized_matrix3 = sample_matrix_normalized.copy()
        optimize_random(u_rand, v_rand, normalized_matrix3, nonzero_indices)
        opt_rand_rmse = rmse(u_rand, v_rand, normalized_matrix3, nonzero_indices)
        print(i, "rmse randomized", rmse(u_rand, v_rand, normalized_matrix3, nonzero_indices))
        if min_rmse > opt_rand_rmse:
            min_rmse = opt_rand_rmse
            best_u = u_rand.copy()
            best_v = v_rand.copy()
    print("rmse by random ", rmse(best_u, best_v, normalized_matrix3, nonzero_indices))

    # STEP 4 Apply Optimization
    # Given the optimization protocols that produced the minimum rmse value, compute the recommendation matrix
    # Compute UV to then add back the averages from normalization
    uv_product = np.matmul(best_u, best_v)
    # Add average column value to each element in UV
    avg_each_col_mat = np.repeat([avg_each_col], uv_product.shape[0], axis=0)
    rec_mat = np.add(uv_product, avg_each_col_mat)

    # Add average row value to each element in UV
    avg_each_row_mat = np.repeat([avg_each_row], uv_product.shape[1], axis=0).transpose()
    rec_mat = np.add(rec_mat, avg_each_row_mat)

    print(sample_matrix.toarray())
    print(rec_mat)

    print("Clip Matrix")
    rec_mat = np.clip(rec_mat, 0, 9)
    print(rec_mat)

    # JordiKai Messing Around
    recommendations = []
    for coord in nonzero_indices:
        book = coord[0]
        user = coord[1]
        # print(coord)
        # print(rec_mat[book, user])
        recommendations.append([coord, rec_mat[book, user]])
    # print(recommendations)

if __name__ == "__main__": main()
