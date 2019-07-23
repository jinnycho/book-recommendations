import math
import heapq
import random
import time
import pickle
import csv
import uvd
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix
from random import randint
from create_util_mat import UtilMat
from collections import defaultdict

def conf_leq(rating):
    return (100.0/3)*rating - (100.0/3)


def conf_greater_than(rating):
    return 20.0*rating


def convert_predictions_to_binary(prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine):
    """ Takes the original prediction dictionaries and outputs binary prediction dictionaries based on the ratings. A rating of less than or equal to 2.5 is a 0, bad book, whereas a rating of greater than 2.5 is a 1, good book.
    At the same time it calculates the confidence for a given rating. We've determined the two piecewise functions and wrote them as helper funct."""
    binary_dict_euclidean = {}
    binary_dict_cosine = {}
    binary_dict_adjusted_cosine = {}
    euclidean_confidence_dict = {}
    cosine_confidence_dict = {}
    adjusted_cosine_confidence_dict = {}
    for user in prediction_dict_euclidean.keys():
        binary_dict_euclidean[user] = {}
        binary_dict_cosine[user] = {}
        binary_dict_adjusted_cosine[user] = {}

        euclidean_confidence_dict[user] = {}
        cosine_confidence_dict[user] = {}
        adjusted_cosine_confidence_dict[user] = {}

        for book in prediction_dict_euclidean[user].keys():
            if prediction_dict_euclidean[user][book] > 2.5:
                binary_dict_euclidean[user][book] = 1
                euclidean_confidence_dict[user][book] = conf_greater_than(prediction_dict_euclidean[user][book])
            else:
                binary_dict_euclidean[user][book] = 0
                euclidean_confidence_dict[user][book] = conf_leq(prediction_dict_euclidean[user][book])

            if prediction_dict_cosine[user][book] > 2.5:
                binary_dict_cosine[user][book] = 1
                cosine_confidence_dict[user][book] = conf_greater_than(prediction_dict_cosine[user][book])
            else:
                binary_dict_cosine[user][book] = 0
                cosine_confidence_dict[user][book] = conf_leq(prediction_dict_cosine[user][book])

            if prediction_dict_adjusted_cosine[user][book] > 2.5:
                binary_dict_adjusted_cosine[user][book] = 1
                adjusted_cosine_confidence_dict[user][book] = conf_greater_than(prediction_dict_adjusted_cosine[user][book])
            else:
                binary_dict_adjusted_cosine[user][book] = 0
                adjusted_cosine_confidence_dict[user][book] = conf_leq(prediction_dict_adjusted_cosine[user][book])

    return binary_dict_euclidean, binary_dict_cosine, binary_dict_adjusted_cosine, euclidean_confidence_dict, cosine_confidence_dict, adjusted_cosine_confidence_dict


def data_for_hybrid(wtr, prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine, binary_dict_euclidean, binary_dict_cosine, binary_dict_adjusted_cosine, euclidean_confidence_dict, cosine_confidence_dict, adjusted_cosine_confidence_dict, k):
    """ In order to analyze CF's results we need to store the following information in a csv
    Each of these will be a column in the csv
    - the actual rating for a given book
    - the predicted rating for a book (binary)
        A book can be classified as either good or bad. A good book has a rating of 1 and a bad book has a rating of 0
    - confidence the predicted rating is going to be a good book
        the confidence is a percentage
    - the number of training ratings used to generate that prediction
    """
    # Load actual ratings from dev_dict (eval_dict)
    # dev_dict = pickle.load(open("datafile/dev_dict.p", "rb"))
    dev_dict = pickle.load(open("datafile/eval_dict.p", "rb"))

    for user, value in prediction_dict_euclidean.items():
        # all three sim dicts should have same user book info
        for book, rating in value.items():
            actual_rating = dev_dict[user][book]

            # Euclidean
            euc_predicted_rating = prediction_dict_euclidean[user][book]
            euc_confidence = euclidean_confidence_dict[user][book]
            euc_binary_prediction_rating = binary_dict_euclidean[user][book]

            euc_row_to_write = [user, book, actual_rating, euc_predicted_rating, "euclidean", euc_confidence, k, euc_binary_prediction_rating]
            wtr.writerow(euc_row_to_write)


            # Cosine
            cos_predicted_rating = prediction_dict_cosine[user][book]
            cos_confidence = cosine_confidence_dict[user][book]
            cos_binary_prediction_rating = binary_dict_cosine[user][book]

            cos_row_to_write = [user, book, actual_rating, cos_predicted_rating, "cosine", cos_confidence, k, cos_binary_prediction_rating]
            wtr.writerow(cos_row_to_write)


            # Adjusted cosine
            adj_predicted_rating = prediction_dict_adjusted_cosine[user][book]
            adj_confidence = adjusted_cosine_confidence_dict[user][book]
            adj_binary_prediction_rating = binary_dict_adjusted_cosine[user][book]

            adj_row_to_write = [user, book, actual_rating, adj_predicted_rating, "adjusted cosine", adj_confidence, k, adj_binary_prediction_rating]
            wtr.writerow(adj_row_to_write)



def main():
    # filename = 'datafile/dev_cf_data_hybrid.csv'
    filename = 'datafile/eval_cf_data_hybrid.csv' #Best accuracy pt for CF was k=30, cosine, unnorm
    wtr = csv.writer(open (filename, 'w'), delimiter=',', lineterminator='\n')

    col_heading = ["user_indx", "book_indx", "actual_rating", "predicted_rating", "similarity_metric", "confidence", "k", "binary_prediction"]
    wtr.writerow(col_heading)

    # For evaluation
    k = 30
    euc_filename = "eval_unnorm_pred/prediction_dict_euclidean_" + str(k) + ".p"
    cos_filename = "eval_unnorm_pred/prediction_dict_cosine_" + str(k) + ".p"
    adj_cos_filename = "eval_unnorm_pred/prediction_dict_adj_cosine_" + str(k) + ".p"

    prediction_dict_euclidean = pickle.load(open(euc_filename, "rb"))
    prediction_dict_cosine = pickle.load(open(cos_filename, "rb"))
    prediction_dict_adjusted_cosine = pickle.load(open(adj_cos_filename, "rb"))

    print("--- create binary dict ---")
    binary_dict_euclidean, binary_dict_cosine, binary_dict_adjusted_cosine, euclidean_confidence_dict, cosine_confidence_dict, adjusted_cosine_confidence_dict = convert_predictions_to_binary(prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine)

    print("--- write csv files for hybrid weights ---")
    data_for_hybrid(wtr, prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine, binary_dict_euclidean, binary_dict_cosine, binary_dict_adjusted_cosine, euclidean_confidence_dict, cosine_confidence_dict, adjusted_cosine_confidence_dict, k)

    pickle.dump(adjusted_cosine_confidence_dict, open("datafile/cf_confidence_dict_eval.p", "wb")) #saves the confidence dict for adj_cos with k=30 on

    """
    # For development
    for k in range(2, 31):
        if k != 30:
            continue
        print("--- k = ", k, " ---")

        euc_filename = "datafile/prediction_dict_euclidean_" + str(k) + ".p"
        cos_filename = "datafile/prediction_dict_cosine_" + str(k) + ".p"
        adj_cos_filename = "datafile/prediction_dict_adj_cosine_" + str(k) + ".p"

        prediction_dict_euclidean = pickle.load(open(euc_filename, "rb"))
        prediction_dict_cosine = pickle.load(open(cos_filename, "rb"))
        prediction_dict_adjusted_cosine = pickle.load(open(adj_cos_filename, "rb"))

        print("--- create binary dict ---")
        binary_dict_euclidean, binary_dict_cosine, binary_dict_adjusted_cosine, euclidean_confidence_dict, cosine_confidence_dict, adjusted_cosine_confidence_dict = convert_predictions_to_binary(prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine)

        # if k == 30:
        #     pickle.dump(cosine_confidence_dict, open("datafile/cf_confidence_dict.p", "wb"))
        print("--- write csv files for hybrid weights ---")
        data_for_hybrid(wtr, prediction_dict_euclidean, prediction_dict_cosine, prediction_dict_adjusted_cosine, binary_dict_euclidean, binary_dict_cosine, binary_dict_adjusted_cosine, euclidean_confidence_dict, cosine_confidence_dict, adjusted_cosine_confidence_dict, k)
    """
    wtr.close()


if __name__ == "__main__": main()
