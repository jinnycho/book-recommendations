""" Authors: Shatian Wang, Eunjin (Jinny) Cho, and JordiKai Watanabe-Inouye
Collaborative Filtering with Item-to-Item :  Implementing k Nearest Neighbor algorithm
References:
    [1]http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
    [2]http://stackoverflow.com/questions/36557472/calculate-the-euclidean-distance-in-scipy-csr-matrix """

import math
import heapq
import random
import time
import pickle
import scipy.sparse as sp
import numpy as np
from create_util_mat import UtilMat
from scipy.sparse import csr_matrix
from scipy.spatial import distance


def save_dict(prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine, euclidean_filename, cosine_filename, adjusted_cosine_filename):
    pickle.dump(prediction_dict_euclidean, open(euclidean_filename, "wb"))
    pickle.dump(prediction_dict_cosine, open(cosine_filename, "wb"))
    pickle.dump(prediction_dict_adjusted_cosine, open(adjusted_cosine_filename, "wb"))
    print ("-------- saved predicted dicts --------")


def create_sample_matrix(n, m, p):
    """
    Used in initial development and testing of code, this method creates a sample csr matrix with a specified sparsity level
    :param n: number of rows in the matrix subset
    :param m: number of columns in the matrix subset
    :param p: the percentage of sparsity in the matrix
    :return: a subset of the matrix for testing purposes
    """
    zeros_in_ratings_list = (math.ceil((9*p)/(1 - p)))*[0]
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


def user_avg_rating(matrix):
    """Given the original matrix, store the average rating for a given user in a
    list."""
    print("--- creating user avg rating lst ---")
    m = matrix.toarray()
    user_avg_rating_lst = m.sum(0)/(m != 0).sum(0)
    return user_avg_rating_lst


def averages(matrix):
    matrix = matrix.transpose()
    (r,c,d)=sp.find(matrix)
    countings=np.bincount(r)
    sums=np.bincount(r,weights=d)
    averages = [] #an array storing the average of each row (book)
    for i in range(len(countings)):
        if countings[i] == 0:
            averages.append(0)
        else:
            averages.append(sums[i]/countings[i])
    return averages


def normalize(matrix):
    """Normalize book's ratings by subtracting out each books's average rating"""
    # find the average of the rows
    '''
    r: the row coordinate, c: the column coordinate
    r and c are both given as arrays, reading the coorsponding positions as tuples
    will provide the coord in the matrix with non-zero entries
    '''
    (r,c,d)=sp.find(matrix)
    countings=np.bincount(r)
    sums=np.bincount(r,weights=d)
    averages = [] #an array storing the average of each row (book)
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
    # and the array of row averages
    return (matrix - diag_averages*matrix_copy), averages


def add(normalized_matrix, track_rated):
    # adding 0.001 to rated books to ensure cosine sim is not zero
    normalized_matrix_copy = normalized_matrix.copy()
    for user in track_rated.keys():
        books = track_rated[user]
        for b in books:
            normalized_matrix_copy[b,user] += 0.001
    return normalized_matrix_copy


def prediction(prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine, N, training_matrix):
    # grabs dicts from array at kth pos
    """ final_prediction is an array that holds three prediction dicts (euc, cos, adj cos) in each spot.
    The index represents the number of neighbors considered when predicting ratings (k = index + 2).
    For instance 0th index in the list contains the list [k=2 euc, k=2 cos, k=2 cos-a]"""
    final_prediction = []
    for k in range(2, 31):
        final_prediction_dict_euclidean = {}
        final_prediction_dict_cosine = {}
        final_prediction_dict_adjusted_cosine = {}
        for user, value in N.items():
            final_prediction_dict_euclidean[user] = {}
            final_prediction_dict_cosine[user] = {}
            final_prediction_dict_adjusted_cosine[user] = {}
            for book, nearest_list in value.items():
                # ---------- Euclidean ----------
                # it means it's "user_average" case
                if book in prediction_dict_euclidean[user]:
                    final_prediction_dict_euclidean[user][book] = "user_average"
                else:
                    similarity_sum = 0.0
                    voted_sum = 0.0
                    for neighbor in nearest_list[0][:k]:
                        dist = neighbor[0]
                        sim = 1.0 / (1.0 + dist)  # convert from distance to similarity
                        similarity_sum += abs(sim)
                        voted_sum += (sim * training_matrix[neighbor[1], user])  # sum(similarity * existing rating)
                    if similarity_sum == 0:
                        final_prediction_dict_euclidean[user][book] = "user_average"
                    else:
                        prediction = voted_sum / similarity_sum
                        final_prediction_dict_euclidean[user][book] = prediction

                # ---------- Cosine ----------
                # it means it's "user_average" case
                if book in prediction_dict_cosine[user]:
                    final_prediction_dict_cosine[user][book] = "user_average"
                else:
                    similarity_sum = 0.0
                    voted_sum = 0.0
                    for neighbor in nearest_list[1][:k]:
                        sim = neighbor[0]
                        similarity_sum += abs(sim)
                        voted_sum += (sim * training_matrix[neighbor[1], user])  # sum(similarity * existing rating)
                    if similarity_sum == 0:
                        prediction = "user_average"
                    else:
                        prediction = voted_sum / similarity_sum
                    final_prediction_dict_cosine[user][book] = prediction

                # ---------- Adjusted Cosine ----------
                # it means it's "user_average" case
                if book in prediction_dict_adjusted_cosine[user]:
                    final_prediction_dict_adjusted_cosine[user][book] = "user_average"
                else:
                    similarity_sum = 0.0
                    voted_sum = 0.0
                    for neighbor in nearest_list[2][:k]:
                        sim = abs(neighbor[0])
                        if math.isnan(sim):
                            continue
                        similarity_sum += sim
                        voted_sum += (sim * training_matrix[neighbor[1], user])  # sum(similarity * existing rating)
                    if similarity_sum == 0:
                        prediction = "user_average"
                    else:
                        prediction = voted_sum / similarity_sum
                    final_prediction_dict_adjusted_cosine[user][book] = prediction

        final_prediction.append([final_prediction_dict_euclidean, final_prediction_dict_cosine, final_prediction_dict_adjusted_cosine])
    return final_prediction


def adjusted_cos_similarity(book_vector_1, book_vector_2, corated_users, user_avg_rating_lst):
    """ Calculates the adjusted cosine similarity between two co-rated books. Which allows for the differences in rating scales between different users to be taken into account - by subtracting the user's average we are normalizing the rating. NOTE that book_vector_2 is a co-rated book of book_vector_1.

    Adjusted cosine similarity is defined as the product of the difference between the rated book b_1 and the average rating for u and the difference between the rated book b_2 and the average rating for u """
    numerator = 0.0
    denominator_b1 = 0.0
    denominator_b2 = 0.0
    for i in range(len(book_vector_1)):
        corated_user = corated_users[i]
        # note that the len of book_vector_2 is equal to book_vector_1
        numerator += (book_vector_1[i] - user_avg_rating_lst[corated_user])*(book_vector_2[i] - user_avg_rating_lst[corated_user])
        denominator_b1 += (book_vector_1[i] - user_avg_rating_lst[corated_user])**2
        denominator_b2 += (book_vector_2[i] - user_avg_rating_lst[corated_user])**2
    denominator = (math.sqrt(denominator_b1)*math.sqrt(denominator_b2))
    adjusted_cosine_similarity = numerator/denominator
    return adjusted_cosine_similarity


def corated_books(book, rated_book, training_matrix, nonzero_training_indices_b2u):
    """ Construct two vectors b1 and b2 that consist of co-rated items
    Vectors only consist of ratings from users who have rated both book and rated_book.
    Thereby the vectors returned are subvectors of book and rated_book. """
    b1 = []
    b2 = []
    corated_users = []
    b1_users = nonzero_training_indices_b2u[book]
    b2_users = nonzero_training_indices_b2u[rated_book]
    for user in b1_users:
        if user in b2_users:
            corated_users.append(user)
            b1.append(training_matrix[book,user])
            b2.append(training_matrix[rated_book,user])
    return b1, b2, corated_users


def kNN(target_entries_dict, training_matrix, nonzero_training_indices, nonzero_training_indices_b2u, user_avg_rating_lst, k):
    """
    Using the k nearest neighbors for each book, computes a dictionary that
    maps a book to recommended book, rating pairs
    :param target_entries_dict: information for either development or evaluation;
            the keys for this dictionary are users and the values are the books
            that need to be given a rating for either development or evaluation
    :param training_matrix: training matrix with current ratings (NOT normalized)
    :param nonzero_training_indices: information on rated books for user,
            we do not want information on unrated books to influence distance ect.
    :param k: an integer representing the number of neighbors considered for a given book
    :return prediction_dict:
    """
    ''' The prediction dictionary is structured as nested dictionaries. It maps a user_id to a dictionary, which maps book_ids to ratings. Recall that the id's used in this dictionary correspond to the coordinate in the matrix. Specifically the inner dictionary should have at most k ratings, we only consider a given book's top k neighbors, note that a book must have been rated for a given user if it is to be a neighbor '''
    prediction_dict_euclidean = {}
    prediction_dict_cosine = {}
    prediction_dict_adjusted_cosine = {}
    ''' N is a nested dictionary that holds three heaps for some user, book pairing. The length of the heap is 30 unless there are less neighbors. '''
    N = {} # {user1: {book1: [euc_nearest_nei, cos_nearest_nei, euc_nearest_nei], book2: ...}, user2: {...} ...}

    # Track the situations where a book to rate doesn't have any corated books
    no_corated_books = {} # maps user to the number of NOT corated books
    user_count_no_corated_books = 0

    for user in range(98268):
        count_no_corated_books = 0

        books_to_rate = target_entries_dict[user]
        books_rated = nonzero_training_indices[user]

        prediction_dict_euclidean[user] = {}
        prediction_dict_cosine[user] = {}
        prediction_dict_adjusted_cosine[user] = {}
        N[user] = {}
        #Print statements for clarity when running
        if int(user) % 1000 == 0:
            print("User: ", user, "\nNum books to rate: ", len(books_to_rate))
        if len(books_to_rate) > 300:
            print("User: ", user, "\nNum books to rate: ", len(books_to_rate))

        for book in books_to_rate:
            book_heap_euclidean_dist = []
            book_heap_cosine_similarity = []
            book_heap_adjusted_cosine_similarity = []

            #Check if book is zero vector
            book_zero = not np.any(training_matrix.getrow(book).toarray())
            if book_zero:
                prediction_dict_euclidean[user][book] = "user_average"
                prediction_dict_cosine[user][book] = "user_average"
                prediction_dict_adjusted_cosine[user][book] = "user_average"
                N[user][book] = []
                # print("Book is zero vector. Set prediction for book ", book, " as user ", user, " avg = ", user_avg_rating_lst[user])
                continue
            for rated_book in books_rated:
                b1, b2, corated_users = corated_books(book, rated_book, training_matrix, nonzero_training_indices_b2u)

                if len(b1) == 0 and len(b2) == 0:
                    count_no_corated_books += 1
                    continue

                euclidean_dist = distance.euclidean(b1, b2)
                cosine_similarity = 1 - distance.cosine(b1, b2)
                adjusted_cosine_similarity = adjusted_cos_similarity(b1, b2, corated_users, user_avg_rating_lst)

                # each other rated book for a given user is pushed into the book's heap
                heapq.heappush(book_heap_euclidean_dist, (euclidean_dist, rated_book))
                heapq.heappush(book_heap_cosine_similarity, (cosine_similarity, rated_book))
                if not math.isnan(adjusted_cosine_similarity):
                    heapq.heappush(book_heap_adjusted_cosine_similarity, (adjusted_cosine_similarity, rated_book))

            no_corated_books[user] = (count_no_corated_books)
            # if the heap is empty then we need to set the prediction to the avg rating for the given user
            if count_no_corated_books == len(books_rated): #implies that the heaps are empty
                user_count_no_corated_books += 1
                prediction_dict_euclidean[user][book] = "user_average"
                prediction_dict_cosine[user][book] = "user_average"
                prediction_dict_adjusted_cosine[user][book] = "user_average"

            neighbors_list_euclidean = heapq.nsmallest(k, book_heap_euclidean_dist) # smaller euclidean distance means more similar books
            neighbors_list_cosine = heapq.nlargest(k, book_heap_cosine_similarity) # larger cosine means more similar books
            neighbors_list_adjusted_cosine = heapq.nlargest(k, book_heap_adjusted_cosine_similarity) # larger adjusted cosine means more similar books

            # Create N
            N[user][book] = [neighbors_list_euclidean, neighbors_list_cosine, neighbors_list_adjusted_cosine]

    # Calculates & saves the predictions for k = 2 through 30
    final_prediction = prediction(prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine, N, training_matrix)

    print("Edge cases with no co-rated books ", user_count_no_corated_books)
    return final_prediction


def track_rated(matrix):
    """ Given a matrix, determines the coordinates of the books that have been rated
    (all nonzero entries)
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
    :param matrix: the original training matrix, constructed in create_util_mat
    :return rated: a dictionary, whose keys is the user_id and value is a list of book_ids that have been rated
    """
    rated = {}
    nonzero_ndarray = np.transpose(np.nonzero(matrix)) #look at documentation to see why the transpose is taken
    for coord in nonzero_ndarray:
        if coord[1] not in rated.keys():
            rated[coord[1]] = [coord[0]]
        else:
            rated[coord[1]].append(coord[0])
    return rated


def track_rated_books(matrix):
    """maps book to users who've rated book"""
    rated = {}
    nonzero_ndarray = np.transpose(np.nonzero(matrix)) #look at documentation to see why the transpose is taken
    for coord in nonzero_ndarray:
        if coord[0] not in rated.keys():
            rated[coord[0]] = [coord[1]]
        else:
            rated[coord[0]].append(coord[1])
    return rated


def main():
    # Load data structures
    util_mat_obj = pickle.load(open("datafile/UtilMat_obj.p", "rb"))
    training_matrix = util_mat_obj.new_matrix
    user_index_to_id_dict = util_mat_obj.index_to_user_dict
    index_to_isbn_dict = util_mat_obj.index_to_isbn_dict
    print ("-------- loaded matrix--------")

    dev_dict = pickle.load(open("datafile/dev_dict.p", "rb"))
    # dev_dict = pickle.load(open("datafile/eval_dict.p", "rb"))

    # Track nonzero entries
    nonzero_training_indices = track_rated(training_matrix)
    nonzero_training_indices_b2u = track_rated_books(training_matrix)

    normalized_matrix, averages = normalize(training_matrix)
    normalized_matrix = add(normalized_matrix, nonzero_training_indices)

    # Track average rating for each user
    # user_avg_rating_lst = user_avg_rating(training_matrix)
    # pickle.dump(user_avg_rating_lst, open("datafile/user_avg_rating_lst.p", "wb"))
    user_avg_rating_lst = pickle.load(open("datafile/user_avg_rating_lst.p", "rb"))

    # Test for k = 30
    start_time = time.time()
    print("--- calculate kNN with training_matrix ---")
    final_prediction_unnorm = kNN(dev_dict, training_matrix, nonzero_training_indices, nonzero_training_indices_b2u, user_avg_rating_lst, 30)
    final_prediction_norm = kNN(dev_dict, normalized_matrix, nonzero_training_indices, nonzero_training_indices_b2u, user_avg_rating_lst, 30)

    print ("-------- predicted matrix--------")

    # with the original training matrix
    # saves the predictions for k = 2 through 30
    k = 2
    for prediction in final_prediction_unnorm:
        for user in range(98268):
            for book in dev_dict[user]:

                if prediction[0][user][book] == "user_average":
                    prediction[0][user][book] = user_avg_rating_lst[user]
                if prediction[1][user][book] =="user_average":
                    prediction[1][user][book] = user_avg_rating_lst[user]
                if prediction[2][user][book] =="user_average":
                    prediction[2][user][book] = user_avg_rating_lst[user]

                if prediction[0][user][book] < 1.0:
                    prediction[0][user][book] = 1.0
                if prediction[0][user][book] > 5.0:
                    prediction[0][user][book] = 5.0
                if prediction[1][user][book] < 1.0:
                    prediction[1][user][book] = 1.0
                if prediction[1][user][book] > 5.0:
                    prediction[1][user][book] = 5.0
                if prediction[2][user][book] < 1.0:
                    prediction[2][user][book] = 1.0
                if prediction[2][user][book] > 5.0:
                    prediction[2][user][book] = 5.0

        # euc_pred_filename = "eval_pred/prediction_dict_euclidean_" + str(k) + ".p"
        # cos_pred_filename = "eval_pred/prediction_dict_cosine_" + str(k) + ".p"
        # a_cos_pred_filename = "eval_pred/prediction_dict_adj_cosine_" + str(k) + ".p"

        euc_pred_filename = "dev_unnorm_pred/prediction_dict_euclidean_" + str(k) + ".p"
        cos_pred_filename = "dev_unnorm_pred/prediction_dict_cosine_" + str(k) + ".p"
        a_cos_pred_filename = "dev_unnorm_pred/prediction_dict_adj_cosine_" + str(k) + ".p"

        pickle.dump(prediction[0], open(euc_pred_filename, "wb"))
        pickle.dump(prediction[1], open(cos_pred_filename, "wb"))
        pickle.dump(prediction[2], open(a_cos_pred_filename, "wb"))

        k += 1

    # With normalized matrix
    # saves the predictions for k = 2 through 30
    k = 2
    for prediction in final_prediction_norm:
        for user in range(98268):
            for book in dev_dict[user]:

                if prediction[0][user][book] == "user_average" or prediction[1][user][book] =="user_average" or prediction[2][user][book] =="user_average":
                    if prediction[0][user][book] == "user_average":
                        # print("first if check")
                        prediction[0][user][book] = user_avg_rating_lst[user]
                    if prediction[1][user][book] =="user_average":
                        # print("second if check")
                        prediction[1][user][book] = user_avg_rating_lst[user]
                    if prediction[2][user][book] =="user_average":
                        # print("thrid if check")
                        prediction[2][user][book] = user_avg_rating_lst[user]
                else:
                    prediction[0][user][book] -= 0.001 # undo + 0.001
                    prediction[0][user][book] += averages[book] # denormalization

                    prediction[1][user][book] -= 0.001 # undo + 0.001
                    prediction[1][user][book] += averages[book] # denormalization

                    prediction[2][user][book] -= 0.001 # undo + 0.001
                    prediction[2][user][book] += averages[book] # denormalization

                if prediction[0][user][book] < 1.0:
                    prediction[0][user][book] = 1.0
                if prediction[0][user][book] > 5.0:
                    prediction[0][user][book] = 5.0
                if prediction[1][user][book] < 1.0:
                    prediction[1][user][book] = 1.0
                if prediction[1][user][book] > 5.0:
                    prediction[1][user][book] = 5.0
                if prediction[2][user][book] < 1.0:
                    prediction[2][user][book] = 1.0
                if prediction[2][user][book] > 5.0:
                    prediction[2][user][book] = 5.0

        # euc_pred_filename = "eval_normalized_pred/prediction_dict_euclidean_" + str(k) + ".p"
        # cos_pred_filename = "eval_normalized_pred/prediction_dict_cosine_" + str(k) + ".p"
        # a_cos_pred_filename = "eval_normalized_pred/prediction_dict_adj_cosine_" + str(k) + ".p"

        euc_pred_filename = "dev_norm_pred/prediction_dict_euclidean_" + str(k) + ".p"
        cos_pred_filename = "dev_norm_pred/prediction_dict_cosine_" + str(k) + ".p"
        a_cos_pred_filename = "dev_norm_pred/prediction_dict_adj_cosine_" + str(k) + ".p"

        pickle.dump(prediction[0], open(euc_pred_filename, "wb"))
        pickle.dump(prediction[1], open(cos_pred_filename, "wb"))
        pickle.dump(prediction[2], open(a_cos_pred_filename, "wb"))

        k += 1
    print("--- Execution Time ---\n %s " % (time.time() - start_time))


if __name__ == "__main__": main()
