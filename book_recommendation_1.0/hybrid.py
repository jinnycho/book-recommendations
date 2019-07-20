'''
Program to get prediction probabilities for CB, CF, and Hybrid Recommender

Authors: Ryan Gorey, JordiKai Watanabe, Sofia Serrano, Jinny Cho, Shatian Wang

This program gets the probability a user rates a book with a high score for
all user/book combinations in a dataset. The program saves the probabilities the
CB and CF system, as well as the actual rating. This data can be used to then
calculate the actual rating the hybrid system would have given the user/book
combo, which will be calculated and evaluated in an R Markdown script.

Uses offline processing - the probabilities between CB and CF system will already
be calculated and will just be fetched with this system. (More of a data processor
at this phase of the project given that we aren't actually making recommendations)
'''

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
import datetime

isbn_to_index_dict = None
user_to_index_dict = None


def get_hybrid_predictions(cf_prob_dict, cb_filename_extension):
    """ Calculates and stores in a csv the probabilities a user rates a
    book positively given a CB and CF framework. Given the accuracy of both
    algorithms weight accordingly to produce a prediction for a user-book
    The predictions are classifications of books as either good or bad.

    Input:
    :actual_dict: a nested dictionary with users as keys and values as
    dictionaries whose keys are books and values are actual ratings
    a user has given a book.

    :cf_prob_dict: a nested dictionary with users as keys and values as
    dictionaries whose keys are books and values are the probability that
    the user has rated the book highly.

    :cb_filename_extension: string extension for cb information to use
    """
    # TODO: Make new csv file
    date = str(datetime.datetime.now())
    day = date[6:10]
    time = date[11:16]
    time = time[0:2] + "-" + time[3:5]
    results = open("datafile/hybrid_results_" + str(day) + "_" + str(time) +
                    ".csv", "w")
    results.write("num_training_books,num_good_training_books,cf_prob_good," +
                    "cb_prob_good,actual_rating_nb\n")

    cb_ratings = open("datafile/results_by_ratings_" + cb_filename_extension, "r")
    user_file = open("datafile/results_by_users_" + cb_filename_extension, "r")
    cb_ratings.readline()
    user_file.readline()
    cur_user_info = user_file.readline().split(",")

    for line in cb_ratings:
        line_info = line.split(",")
        user_id = line_info[0].strip()
        isbn = line_info[1].strip()
        while isbn.startswith("0"):
            isbn = isbn[1:]
        actual_rating_nonbinary = line_info[2].strip()
        cb_good_prob = line_info[7].strip()

        if cur_user_info[0] != user_id:
            cur_user_info = user_file.readline().split(",")
        num_training_books = cur_user_info[2]
        num_good_t_books = cur_user_info[1]

        user_ind = user_to_index_dict[user_id]
        book_ind = isbn_to_index_dict[isbn]
        cf_good_prob = cf_prob_dict[user_ind][book_ind]

        results.write(num_training_books + "," + num_good_t_books + "," +
            str(cf_good_prob) + "," + cb_good_prob + "," +
            actual_rating_nonbinary + "\n")

    results.close()
    user_file.close()
    cb_ratings.close()

    # CF TODO: find pred dict that gives max accuracy


def main():
    util_mat_obj = pickle.load(open("datafile/UtilMat_obj.p", "rb"))
    user_index_to_id_dict = util_mat_obj.index_to_user_dict
    index_to_isbn_dict = util_mat_obj.index_to_isbn_dict
    global isbn_to_index_dict, user_to_index_dict
    isbn_to_index_dict = util_mat_obj.isbn_to_index_dict
    user_to_index_dict = util_mat_obj.user_to_index_dict

    # cf_prob_dict = pickle.load(open("datafile/cf_confidence_dict.p", "rb")) #dev_test
    cf_prob_dict = pickle.load(open("datafile/cf_confidence_dict_eval.p", "rb"))
    cb_extension = "maxent_bothbytfidf_100feats_unpairedfeats_2-23_12-52.csv"

    get_hybrid_predictions(cf_prob_dict, cb_extension)


if __name__ == "__main__": main()
