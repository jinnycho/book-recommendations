"""
user_profile_learner.py
Author: Sofia Serrano
Learns a user profile from content-based models of books and ratings,
and then uses that profile to predict ratings.
Options for learning user profile:
    - naive Bayes classifier
    - MaxEnt

To run full test using 100 features max, run as follows:
python3 user_profile_learner.py {tfidf, sentiment, both_indep, both_by_tfidf} {naive_bayes, maxent} 100

Downloads required:
    - sklearn.linear model
        (download from http://scikit-learn.org/stable/install.html)
    - scipy.sparse
    - numpy
Files required:
    - training ratings by users, provided in variable at top of file
      called training_data_filename
    - ratings by users to train classifier to match, provided
      in evaluation_data_filename
    - datafile/isbns_to_indices_dict.txt
    - datafile/indices_to_isbns_dict.txt
    - datafile/TFIDF.npz
    - datafile/Sentiment.npz
    - datafile/ISBNNumsInCommon.txt
    - datafile/narrowed_dict.txt
    - datafile/fewer_nouns.txt
"""
from math import pow
from math import sqrt
from math import pi
from math import e
from math import log
from sys import exit
from sys import argv
from feature_collector import FeatureCollector
from feature_collector import mean_of_list
from max_ent_weight_trainer import MaxEntWeightTrainer
from max_ent_weight_trainer import unlog_log_probs
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import ast
import datetime
from sklearn.linear_model import LogisticRegression


MAX_NUM_FEATURES = 100 # will be overwritten by command-line parameter
MAX_NUM_FEATURE_CANDIDATES = 200

USE_PAIRED_FEATURES_MAXENT = False # False works better for MaxEnt
USE_PAIRED_FEATURES_NAIVE_BAYES = True # True works better for Naive Bayes
BINARY_FEATURES = True # binary-fies real-number-valued vects; *does not work* currently if set to False
SCALE_NONBINARY_MAXENT_FEATURES = True # currently just works for nonnegative features

# for non-package MaxEnt implementation, which has not been updated to work with the current code:
MAX_WEIGHT_TRAINING_ITERATIONS = 2000
STOP_TRAINING_THRESHOLD = pow(10, -16)

training_data_filename = "../datafile/training_ratings.txt"
evaluation_data_filename = "../datafile/evaluation_ratings.txt"

with open("../datafile/isbns_to_indices_dict.txt", "r") as file:
    isbns_to_indices = ast.literal_eval(file.readline())
with open("../datafile/indices_to_isbns_dict.txt", "r") as file:
    indices_to_isbns = ast.literal_eval(file.readline())

num_books = 0
f = open("../datafile/ISBNNumsInCommon.txt", "r")
for line in f:
    if line.strip() != "":
        num_books += 1
f.close()
modeled_book_lil_matrix = None
modeled_book_lil_matrix_2 = None
list_of_user_rating_tuples_training = None
list_of_user_rating_tuples_eval = None


def get_list_of_modeled_book_lists_for_all_books(tfidf):
    list_of_all_isbns = []
    f = open("../datafile/ISBNNumsInCommon.txt", "r")
    for isbn in f:
        isbn = isbn.strip()
        if isbn != "":
            list_of_all_isbns.append(isbn)
    f.close()
    return get_list_of_modeled_book_lists(tfidf, list_of_all_isbns, True)


def get_list_of_modeled_book_lists(tfidf, list_of_used_isbns, full_matrix):
    """
    Read in book model vectors from a .npz file and return a list of lists with
    iterable nonzero vector content contained inside. Also returns the number of
    distinct terms that are indexed by the first number in any (term index, value) tuple
    in a compressed book vector, and the mapping of the index of a book in this returned
    list of lists to its index in the overall list of books
    :param tfidf: True if modeled books should be TFIDF, False otherwise
    :param list_of_used_isbns: list of string isbns that the user rated
    :return three things:
        list of lists of (termindex, value) tuples (one inner list per book)
        dict mapping the index of a book in the matrix to the index of that book in the
            overall list of all ISBNs
        number of distinct terms in {TFIDF, Sentiment} model (whichever applies)
    """
    try:
        matrix, iterate_through = get_matrix(tfidf)
        num_terms = len(get_word_list(tfidf))
        if full_matrix:
            return matrix, {i:i for i in range(num_books)}, num_terms
    except:
        print("Matrix file not found.")
        exit(1)
    isbn_indices = get_isbn_indices(list_of_used_isbns)
    isbn_tuple_lists = [[] for i in range(len(isbn_indices))]
    counter = 0
    abridged_matrix = lil_matrix((len(isbn_indices), num_terms))
    matrix_ind_to_book_ind_dict = {}
    for i in range(len(isbn_indices)):
        book_ind = isbn_indices[i]
        matrix_ind_to_book_ind_dict[i] = book_ind
        for j in range(num_terms):
            if matrix[book_ind, j] != 0:
                abridged_matrix[i, j] = matrix[book_ind, j]
    abridged_matrix = csr_matrix(abridged_matrix)
    return abridged_matrix, matrix_ind_to_book_ind_dict, num_terms


def make_list_of_user_rating_tuples_from_file(filename):
    """
    Reads list of users' isbn-rating tuples from the given file and returns two things:
    a list of the user IDs that were covered, and a list of the corresponding bookindex-rating
    tuple lists
    :param filename: file where we can find users' ratings
    :return: 2 things:
        a list of the user IDs in the file, in the order they appear
        a list of lists of (book index, rating) tuples, where each inner list is
            a single user's ratings from this file
    """
    try:
        f = open(filename, "r")
    except:
        print("File " + filename + " not found.")
        exit(1)
    list_of_users = []
    list_of_their_rating_tuples = []
    for line in f:
        if line.strip() != "":
            user_id = line[0: line.index(":")].strip()
            rest_of_line = ""
            if len(line) - 1 >= line.index(":") + 1:
                rest_of_line = line[line.index(":") + 1:].strip()
            ratings = rest_of_line.split(";")
            ratings = ratings[:len(ratings) - 1] # strips out last thing, which is always ''
            rating_tuple_list = []
            for rating in ratings:
                rating = rating.strip()
                isbn_string = rating[1: rating.index(",")]
                int_rating = int(rating[rating.index(",") + 2 : len(rating) - 1])
                rating_tuple_list.append((isbns_to_indices[isbn_string], int_rating))
            list_of_users.append(user_id)
            list_of_their_rating_tuples.append(rating_tuple_list)
    f.close()
    return list_of_users, list_of_their_rating_tuples


def binarify_ratings_by_average(isbn_rating_tuple_list):
    """
    Flattens the list of ratings down to a list where the rating in each tuple is a binary
    good-bad rating (where the maximum bad rating is the average rating the user gave in this set)
    :param isbn_rating_tuple_list: a list of (isbn, rating) tuples
    :return: a new list of (isbn, rating) tuples where each rating is either 0 (bad) or 1 (good)
    """
    avg = 0
    for isbn_rating in isbn_rating_tuple_list:
        avg += isbn_rating[1]
    if len(isbn_rating_tuple_list) != 0:
        avg /= len(isbn_rating_tuple_list)
    return avg, binarify_ratings_with_max_bad_rating(avg, isbn_rating_tuple_list)


def binarify_ratings_with_max_bad_rating(max_bad, isbn_rating_tuple_list):
    """
    Flattens the list of ratings down to a list where the rating in each tuple is a binary
    good-bad rating
    :param max_bad: the maximum rating that is considered bad (assumed to be an integer)
    :param isbn_rating_tuple_list: a list of (isbn, rating) tuples
    :return: a new list of (isbn, rating) tuples where each rating is either 0 (bad) or 1 (good)
    """
    new_list = []
    for rating in isbn_rating_tuple_list:
        if rating[1] <= max_bad:
            new_list.append((rating[0], 0))
        else:
            new_list.append((rating[0], 1))
    return new_list


def std_dev(keyword_mean, keyword_count_list, tfidf):
    sum_of_squared_diffs = 0
    for keyword_count in keyword_count_list:
        sum_of_squared_diffs += pow((keyword_count - keyword_mean), 2)
    sum_of_squared_diffs = sum_of_squared_diffs / len(keyword_count_list)
    keyword_std_dev = sqrt(sum_of_squared_diffs)
    if keyword_std_dev == 0:
        if tfidf:
            return .05 # was previously .0001
        else:
            return 1
    return keyword_std_dev


def get_isbn_indices(list_of_isbns):
    """
    Gets the numerical indices of ISBN numbers in our data
    :param list_of_sorted_isbns: sorted list of string representations of ISBN numbers
    :return: the list of int indices corresponding to the ISBNs
    """
    indices = []
    for isbn in list_of_isbns:
        indices.append(isbns_to_indices[isbn])
    return indices


def get_word_list(tfidf):
    """
    Returns the corresponding dictionary in list form
    :param tfidf: True if dictionary for TFIDF should be returned, False if dictionary
    for Sentiment should be returned
    :return: the list of words
    """
    if tfidf:
        words = open("../datafile/narrowed_dict.txt", "r")
    else:
        words = open("../datafile/fewer_nouns.txt")
    list_to_ret = []
    for word in words:
        word = word.strip()
        if word != "":
            list_to_ret.append(word)
    return list_to_ret


def get_matrix(tfidf):
    """
    Returns the correct sparse matrix of data (as zipped coo_matrix)
    :param tfidf: True if TFIDF matrix requested, False if Sentiment matrix requested
    :return: the loaded matrix as a coo_matrix, and number of nonzero cells in it
    """
    if tfidf:
        loader = np.load("../datafile/TFIDF.npz")
    else:
        loader = np.load("../datafile/Sentiment.npz")
    matrix = lil_matrix(csr_matrix((loader['data'], loader['indices'], loader['indptr'] ),
                         shape = loader['shape']))
    return matrix, len(loader['data'])


def print_sentiment_analysis_vector_summary(isbn, max_words_printed):
    """
    Prints an easily-readable summary of a book's sentiment analysis vector
    :param isbn: string version of the ISBN to print results for
    :param max_words_printed: the max number of words to print under each category
    :return: None (just prints)
    """
    book_index = isbns_to_indices[isbn]
    sentiment_matrix, num_nonzero_elements = get_matrix(False)
    sentiment_words = get_word_list(False)
    good_words = []
    good_scores = []
    bad_words = []
    bad_scores = []
    for i in range(len(sentiment_words)):
        if sentiment_matrix[book_index, i] > 0:
            good_words.append(sentiment_words[i])
            good_scores.append(sentiment_matrix[book_index, i])
        if sentiment_matrix[book_index, i] < 0:
            bad_words.append(sentiment_words[i])
            bad_scores.append(sentiment_matrix[book_index, i])
    good = sorted(zip(good_scores, good_words), reverse=True)
    good_words = [x for (y, x) in good]
    good_scores = [y for (y, x) in good]
    bad = sorted(zip(bad_scores, bad_words))
    bad_words = [x for (y, x) in bad]
    bad_scores = [y for (y, x) in bad]
    print("****************** GOOD WORDS ******************")
    if max_words_printed >= len(good):
        for i in range(len(good)):
            print(good_words[i] + ", " + str(good_scores[i]))
    else:
        for i in range(max_words_printed):
            print(good_words[i] + ", " + str(good_scores[i]))
    print("\n****************** BAD WORDS ******************")
    if max_words_printed >= len(bad):
        for i in range(len(bad)):
            print(bad_words[i] + ", " + str(bad_scores[i]))
    else:
        for i in range(max_words_printed):
            print(bad_words[i] + ", " + str(bad_scores[i]))


def print_tfidf_vector_summary(isbn, max_words_printed):
    """
    Prints an easily-readable summary of a book's TFIDF vector
    :param isbn: string version of the ISBN to print results for
    :param max_words_printed: the max number of words to print under each category
    :return: None (just prints)
    """
    book_index = isbns_to_indices[isbn]
    assert book_index != None, "No matching ISBN found for " + isbn
    sentiment_matrix, num_nonzero_elements = get_matrix(True)
    sentiment_words = get_word_list(True)
    words = []
    scores = []
    for i in range(len(sentiment_words)):
        if sentiment_matrix[book_index, i] > 0:
            words.append(sentiment_words[i])
            scores.append(sentiment_matrix[book_index, i])
    data = sorted(zip(scores, words), reverse=True)
    words = [x for (y, x) in data]
    scores = [y for (y, x) in data]
    print("****************** WORDS ******************")
    if max_words_printed >= len(data):
        for i in range(len(data)):
            print(words[i] + ", " + str(scores[i]))
    else:
        for i in range(max_words_printed):
            print(words[i] + ", " + str(scores[i]))


class UserProfileLearner:
    classifier_type = "naive_bayes"
    not_enough_info = False # only True if a) the user gave only good ratings or
                            # b) the user gave only bad ratings
    tfidf = None
    placeholders = None
    num_terms = None
    num_terms_second_matrix = None
    select_indep = None
    second_matrix_has_info = False

    # for naive Bayes
    priors = None
    likelihoods = None

    # for MaxEnt
    feature_indices = None
    feature_cutoffs = None # only relevant if we're making the features binary
    feature_weights = None

    external_maxent_model = None

    def __init__(self, naive_bayes_classifier, terms_in_dict, terms_in_dict_2, select_indep, tfidf,
                 second_matrix_has_info):
        self.num_terms = terms_in_dict
        self.num_terms_second_matrix = terms_in_dict_2
        self.select_indep = select_indep
        self.tfidf = tfidf
        self.second_matrix_has_info = second_matrix_has_info
        if naive_bayes_classifier == "naive_bayes":
            self.classifier_type = "naive_bayes"
        elif naive_bayes_classifier == "maxent_nopackage":
            self.classifier_type = "maxent_nopackage"
        else:
            self.classifier_type = "maxent_package"
            self.external_maxent_model = LogisticRegression()

    def train(self, training_pairs):
        ratings_given = training_pairs.keys()
        if 0 not in ratings_given or 1 not in ratings_given or len(training_pairs[0]) == 0 or \
            len(training_pairs[1]) == 0:
            self.not_enough_info = True
            if 0 not in ratings_given or len(training_pairs[0]) == 0:
                self.placeholders = [-100000, log(1)]
            else:
                self.placeholders = [log(1), -100000]
        elif self.classifier_type == "naive_bayes":
            self.calculate_priors(training_pairs.keys())
            fc = FeatureCollector(False, MAX_NUM_FEATURES, MAX_NUM_FEATURE_CANDIDATES,
                                  False, USE_PAIRED_FEATURES_NAIVE_BAYES, self.num_terms, self.num_terms_second_matrix,
                                  modeled_book_lil_matrix, modeled_book_lil_matrix_2, self.select_indep,
                                  self.second_matrix_has_info)
            training_feats, orig_vects, self.feature_cutoffs, self.feature_indices = \
                fc.collect_feature_inds_and_training_feats(training_pairs)
            # now remake training_pairs to include only the features selected
            if not USE_PAIRED_FEATURES_NAIVE_BAYES:
                num_bad_feats = len(self.feature_indices[0])
                indices_of_feats_to_remove = []
                for bad_feat in self.feature_indices[0]:
                    if bad_feat in self.feature_indices[1]:
                        rep_feat_index = self.feature_indices[1].index(bad_feat)
                        indices_of_feats_to_remove.append(rep_feat_index + num_bad_feats)
                        del self.feature_indices[1][rep_feat_index]
                self.feature_indices = self.feature_indices[0] + self.feature_indices[1]
                for book in training_feats:
                    temp_list = book[2]
                    for i in indices_of_feats_to_remove:
                        del temp_list[i]
                    book[2] = temp_list
            bad_list = []
            good_list = []
            # format of book: [rating, num_bad_feats, [list of all feats]]
            for book in training_feats:
                if USE_PAIRED_FEATURES_NAIVE_BAYES:
                    num_distinct_feats = book[1]
                    book_vect = book[2][0: num_distinct_feats]
                else:
                    book_vect = book[2]
                if book[0] == 0:
                    bad_list.append(book_vect)
                else:
                    good_list.append(book_vect)

            training_pairs = {0 : bad_list, 1 : good_list}
            self.calculate_likelihoods(training_pairs)
        elif self.classifier_type == "maxent_nopackage": # if MaxEnt is selected
            print("Non-package implementation of MaxEnt classifier not updated, won't work.")
            print("Quitting now.")
            exit(1)
            fc = FeatureCollector(BINARY_FEATURES, MAX_NUM_FEATURES,
                                  MAX_NUM_FEATURE_CANDIDATES,
                                  SCALE_NONBINARY_MAXENT_FEATURES,
                                  USE_PAIRED_FEATURES_MAXENT,
                                  self.num_terms)
            training_feats, orig_vects, self.feature_cutoffs, self.feature_indices = \
                fc.collect_feature_inds_and_training_feats(training_pairs)
            if not USE_PAIRED_FEATURES_MAXENT:
                num_feats = len(self.feature_indices[0]) + len(self.feature_indices[1])
            else:
                num_feats = 2 * len(self.feature_indices)
            self.initialize_weights(num_feats)
            wt = MaxEntWeightTrainer(self, training_feats, orig_vects, BINARY_FEATURES,
                                     MAX_WEIGHT_TRAINING_ITERATIONS, STOP_TRAINING_THRESHOLD)
            wt.train_weights()
        else: # if package MaxEnt implementation selected
            fc = FeatureCollector(BINARY_FEATURES, MAX_NUM_FEATURES,
                                  MAX_NUM_FEATURE_CANDIDATES,
                                  SCALE_NONBINARY_MAXENT_FEATURES,
                                  USE_PAIRED_FEATURES_MAXENT,
                                  self.num_terms, self.num_terms_second_matrix, modeled_book_lil_matrix,
                                  modeled_book_lil_matrix_2, self.select_indep,
                                  self.second_matrix_has_info)
            training_feats, orig_vects, self.feature_cutoffs, self.feature_indices = \
                fc.collect_feature_inds_and_training_feats(training_pairs)
            if not USE_PAIRED_FEATURES_MAXENT:
                num_feats = len(self.feature_indices[0]) + len(self.feature_indices[1])
            else:
                num_feats = len(self.feature_indices)

            # put together the corresponding sparse matrix of training features
            # and put together labels, one per training book (1 or 0)
            training_books_by_feats = lil_matrix((len(training_feats), num_feats))
            labels = []

            i = 0
            for book in training_feats:
                training_books_by_feats[i, 0:num_feats] = book[2][0:num_feats]
                labels.append(book[0])
                i += 1
            labels = np.array(labels)
            self.external_maxent_model = self.external_maxent_model.fit(training_books_by_feats,
                                                                        labels)

    def predict_rating(self, input_vector):
        if self.not_enough_info:
            return self.placeholders
        if self.classifier_type == "naive_bayes":
            return self.predict_rating_naive_bayes(input_vector)
        elif self.classifier_type == "maxent_nopackage":
            return self.predict_rating_max_ent(input_vector)
        else:
            return self.predict_rating_max_ent_package(input_vector)

    ############### Naive Bayes: ###################

    def calculate_priors(self, training_ratings):
        """
        Based on how often different ratings come up in the user's training data,
        we produce prior probabilities of how likely the user is to give that
        particular rating (saved in self.priors)
        :param training_ratings: list of all ratings given (including repeats)
        :return: None
        """
        # just the count of each 0-1 (0 = bad, 1 = good) rating that user gave
        # rating_count + 1 / (total num training ratings + 2)
        # uses Laplace smoothing
        self.priors = [1, 1]
        for rating in training_ratings:
            self.priors[rating] += 1
        total_num_ratings = len(training_ratings)
        for i in range(2):
            self.priors[i] = self.priors[i] / (total_num_ratings + 2)

    def calculate_likelihoods(self, dict_of_book_vects_by_rating):
        """
        Makes and fills out the matrix of estimated parameters, which will be in the
        following format:
            List with 2 (1 for each rating)
                lists of length [book vector length]
                    with a list of length 2 in each spot (first number is estimated
                    mean of distribution, second number is estimated standard deviation)
        :param dict_of_book_vects_by_rating: a dict with ratings as keys, lists of all book
        vectors that received that rating as values
        :return: nothing (we're just training the model)
        """
        # probability of book given rating
        #
        # suppose we say that the value of any given feature in a book vector (for example,
        # a tf-idf-processed word count for one particular book) follows a Gaussian
        # distribution. Then
        # p(number rep. particular book feature | rating) ~ N(mean, standard deviation)
        # estimate those using MLE (mean word count, for example, in all books given this rating)
        #
        # save estimated parameters in a matrix
        self.likelihoods = []
        vect_of_all_keyword_values = None
        feat_matrix_of_origin_vect = None
        ratings_we_need_to_fill_in_later = []
        num_keyword_vals_in_vector = None
        for rating in range(2):
            books_with_rating = dict_of_book_vects_by_rating[rating]
            self.likelihoods.append([])
            vect_of_feat_values_for_rating = None
            for book_vector in books_with_rating:
                # if this is the first book_vector we've come across
                if vect_of_all_keyword_values is None:
                    vect_of_all_keyword_values = []
                    num_keyword_vals_in_vector = len(book_vector)
                    for i in range(num_keyword_vals_in_vector):
                        vect_of_all_keyword_values.append([])
                    feat_matrix_of_origin_vect = []
                    for i in range(len(self.feature_indices)):
                        feat_matrix_of_origin_vect.append(self.feature_indices[i][0])
                # if this is the first book vector for *this rating* we've come across
                if vect_of_feat_values_for_rating is None:
                    vect_of_feat_values_for_rating = []
                    for i in range(num_keyword_vals_in_vector):
                        vect_of_feat_values_for_rating.append([])
                for word_index in range(num_keyword_vals_in_vector):
                    vect_of_all_keyword_values[word_index].append(book_vector[word_index])
                    vect_of_feat_values_for_rating[word_index].append(book_vector[word_index])
            # vect_of_ratings_keyword_values is now either empty (in which case we put off
            # dealing with it until we have vect_of_all_keyword_values filled out) or we're
            # ready to calculate estimated params
            if vect_of_feat_values_for_rating is None:
                ratings_we_need_to_fill_in_later.append(rating)
                self.likelihoods.append([])
            else:
                vect_of_param_estimations = []
                for i in range(len(vect_of_feat_values_for_rating)):
                    keyword_count_list = vect_of_feat_values_for_rating[i]
                    matrix_of_origin = feat_matrix_of_origin_vect[i]
                    if self.second_matrix_has_info and matrix_of_origin == 0:
                        from_tfidf = True
                    elif self.second_matrix_has_info and matrix_of_origin == 1:
                        from_tfidf = False
                    keyword_mean = mean_of_list(keyword_count_list)
                    if not self.second_matrix_has_info:
                        keyword_std_dev = std_dev(keyword_mean, keyword_count_list, self.tfidf)
                    else:
                        keyword_std_dev = std_dev(keyword_mean, keyword_count_list, from_tfidf)
                    param_pair = [keyword_mean, keyword_std_dev]
                    vect_of_param_estimations.append(param_pair)
                self.likelihoods[rating] += vect_of_param_estimations
        vect_to_copy_into_emptys = []
        # now deal with params we didn't get any data for-- give them the average
        for i in range(len(vect_of_all_keyword_values)):
            # put together the vector of params that we'll copy into all empty ratings' lists
            keyword_count_list = vect_of_all_keyword_values[i]
            matrix_of_origin = feat_matrix_of_origin_vect[i]
            if self.second_matrix_has_info and matrix_of_origin == 0:
                from_tfidf = True
            elif self.second_matrix_has_info and matrix_of_origin == 1:
                from_tfidf = False
            overall_keyword_mean = mean_of_list(keyword_count_list)
            if not self.second_matrix_has_info:
                overall_keyword_std_dev = std_dev(overall_keyword_mean, keyword_count_list, self.tfidf)
            else:
                overall_keyword_std_dev = std_dev(overall_keyword_mean, keyword_count_list, from_tfidf)
            param_pair = [overall_keyword_mean, overall_keyword_std_dev]
            vect_to_copy_into_emptys.append(param_pair)
        for rating in ratings_we_need_to_fill_in_later:
            self.likelihoods[rating] = self.likelihoods[rating - 1] + vect_to_copy_into_emptys

    def predict_rating_naive_bayes(self, input_vector):
        """
        Predicts a rating for a single model vector using the existing trained
        classifier
        :param input_vector: a single model vector
        :return: list of length 2, containing non-normalized log probs for bad and
            good ratings
        """
        # for each possible rating (0 or 1), calculate the following:
        # prior:
        #   prior for this rating, multiply by:
        # likelihood: for each feature in our model of a book:
        #   multiply by N(this book's word count for this word | mean count for this word
        #                                                        in a rating from this user,
        #                                                        standard deviation)
        # now normalize by total probability.
        vect_of_log_probs = []
        for rating in range(2):
            log_prob_for_this_rating = log(self.get_prior_at_ind(rating))
            # log prob because it might be really tiny
            log_prob_for_this_rating += self.get_log_likelihood_for_rating(rating, input_vector)
            vect_of_log_probs.append(log_prob_for_this_rating)
        return vect_of_log_probs

    def get_prior_at_ind(self, rating):
        return self.priors[rating]

    def get_log_likelihood_for_rating(self, rating, vector_ind):
        base_coeff = 1 / sqrt(2 * pi)
        total_log_prob = 0
        num_feats = len(self.feature_indices)
        for index_in_training_vect in range(num_feats):
            index_in_full_vect = self.feature_indices[index_in_training_vect][1]
            matrix = self.feature_indices[index_in_training_vect][0]
            mean = self.likelihoods[rating][index_in_training_vect][0]
            feat_std_dev = self.likelihoods[rating][index_in_training_vect][1]
            coeff = base_coeff / feat_std_dev
            if matrix == 0:
                exp = -1*pow((modeled_book_lil_matrix[vector_ind, index_in_full_vect] - mean), 2)/\
                      (2*pow(feat_std_dev, 2))
            else:
                exp = -1*pow((modeled_book_lil_matrix_2[vector_ind, index_in_full_vect] - mean), 2)/\
                      (2*pow(feat_std_dev, 2))
            prob = coeff*pow(e, exp)
            if prob == 0:
                total_log_prob += -10000
            else:
                total_log_prob += log(prob)
        return total_log_prob

    ############### Max Ent Non-Package: ###################

    def predict_rating_max_ent(self, vector):
        """
        Generates the log probabilities of this book vector getting either a bad or good
        rating using the MaxEnt model
        :param vector: the book vector that needs a rating predicted
        :return: a list of two log probabilities, the first for the book getting a bad
        rating and the second for the book getting a good rating
        """
        num_features = len(self.feature_weights[0] + self.feature_weights[1])
        bad_log_likelihood = 0
        num_bad_features = len(self.feature_weights[0])
        for i in range(num_bad_features):
            if not USE_PAIRED_FEATURES_MAXENT:
                word_index = self.feature_indices[0][i]
            else:
                word_index = self.feature_indices[i]
            weight = self.feature_weights[0][i]
            feature = vector[word_index]
            if BINARY_FEATURES:
                instructions = self.feature_cutoffs[0][i]
                if ((feature <= instructions[0]) and instructions[1]) or \
                   ((feature > instructions[0]) and (not instructions[1])):
                    feature = 1
                else:
                    feature = 0
            bad_log_likelihood += weight * feature
        good_log_likelihood = 0
        i = num_bad_features
        while i < num_features:
            if type(self.feature_indices[0]) is list:
                word_index = self.feature_indices[1][i - num_bad_features]
            else:
                word_index = self.feature_indices[i - num_bad_features]
            weight = self.feature_weights[1][i - num_bad_features]
            feature = vector[word_index]
            if BINARY_FEATURES:
                instructions = self.feature_cutoffs[1][i - num_bad_features]
                if ((feature <= instructions[0]) and instructions[1]) or \
                   ((feature > instructions[0]) and (not instructions[1])):
                    feature = 1
                else:
                    feature = 0
            good_log_likelihood += weight * feature
            i += 1
        bad_unnormalized_prob = pow(e, bad_log_likelihood)
        good_unnormalized_prob = pow(e, good_log_likelihood)
        total = bad_unnormalized_prob + good_unnormalized_prob
        return [log(bad_unnormalized_prob/total), log(good_unnormalized_prob/total)]

    def initialize_weights(self, num_features):
        """
        Initializes features weights
        :param num_features: the total number of features that need weights initialized
        :return:
        """
        self.feature_weights = []
        bad_feature_weights = []
        for i in range(num_features // 2):
            bad_feature_weights.append(0)
        good_feature_weights = []
        for i in range(num_features // 2):
            good_feature_weights.append(0)
        self.feature_weights.append(bad_feature_weights)
        self.feature_weights.append(good_feature_weights)

    ############### Max Ent Package: ###################

    def predict_rating_max_ent_package(self, input_vector_index):
        # first, put together the feature vector
        feat_vector = []
        if USE_PAIRED_FEATURES_MAXENT:
            num_features = len(self.feature_indices)
        else:
            num_features = len(self.feature_indices[0]) + len(self.feature_indices[1])
        if not USE_PAIRED_FEATURES_MAXENT:
            num_bad_features = len(self.feature_indices[0])
            for i in range(num_bad_features):
                word_index = self.feature_indices[0][i][1]
                matrix = self.feature_indices[0][i][0]
                if matrix == 0:
                    feature = modeled_book_lil_matrix[input_vector_index, word_index]
                else:
                    feature = modeled_book_lil_matrix_2[input_vector_index, word_index]
                if BINARY_FEATURES:
                    instructions = self.feature_cutoffs[0][i]
                    if ((feature <= instructions[0]) and instructions[1]) or \
                            ((feature > instructions[0]) and (not instructions[1])):
                        feature = 1
                    else:
                        feature = 0
                feat_vector.append(feature)
            i = num_bad_features
            while i < num_features:
                word_index = self.feature_indices[1][i - num_bad_features][1]
                matrix = self.feature_indices[1][i - num_bad_features][0]
                if matrix == 0:
                    feature = modeled_book_lil_matrix[input_vector_index, word_index]
                else:
                    feature = modeled_book_lil_matrix_2[input_vector_index, word_index]
                if BINARY_FEATURES:
                    instructions = self.feature_cutoffs[1][i - num_bad_features]
                    if ((feature <= instructions[0]) and instructions[1]) or \
                            ((feature > instructions[0]) and (not instructions[1])):
                        feature = 1
                    else:
                        feature = 0
                feat_vector.append(feature)
                i += 1
        else:
            for i in range(num_features):
                word_index = self.feature_indices[i][1]
                matrix = self.feature_indices[i][0]
                if matrix == 0:
                    feature = modeled_book_lil_matrix[input_vector_index, word_index]
                else:
                    feature = modeled_book_lil_matrix_2[input_vector_index, word_index]
                if BINARY_FEATURES:
                    instructions = self.feature_cutoffs[0][i]
                    if ((feature <= instructions[0]) and instructions[1]) or \
                            ((feature > instructions[0]) and (not instructions[1])):
                        feature = 1
                    else:
                        feature = 0
                feat_vector.append(feature)
        feat_vector = np.array(feat_vector).reshape(1, -1)

        # now, predict and return results for that feature vector
        raw_results = self.external_maxent_model.predict_log_proba(feat_vector)
        return [raw_results[0][0], raw_results[0][1]]


def make_test_filenames(data_type, classifier_type):
    if classifier_type == "maxent_package":
        classifier_type = "maxent"
    elif classifier_type == "naive_bayes":
        classifier_type = "nb"
    results_filename = "../datafile/results_by_ratings_" + classifier_type
    user_filename = "../datafile/results_by_users_" + classifier_type
    if data_type == "tfidf":
        results_filename += "_tfidf_"
        user_filename += "_tfidf_"
    elif data_type == "sentiment":
        results_filename += "_sentiment_"
        user_filename += "_sentiment_"
    elif data_type == "both_tfidf_determined":
        results_filename += "_bothbytfidf_"
        user_filename += "_bothbytfidf_"
    elif data_type == "both_indep":
        results_filename += "_bothindep_"
        user_filename += "_bothindep_"
    else:
        print("Invalid data type (" + data_type + ") requested. Quitting now.")
        exit(1)
    if (classifier_type.startswith("n") and not USE_PAIRED_FEATURES_NAIVE_BAYES) or \
            (classifier_type.startswith("m") and not USE_PAIRED_FEATURES_MAXENT):
        max_num_feats = MAX_NUM_FEATURES * 2
    else:
        max_num_feats = MAX_NUM_FEATURES
    results_filename += str(max_num_feats) + "feats_"
    user_filename += str(max_num_feats) + "feats_"
    if (classifier_type.startswith("n") and USE_PAIRED_FEATURES_NAIVE_BAYES) or \
            (classifier_type.startswith("m") and USE_PAIRED_FEATURES_MAXENT):
        feat_descriptor = "pairedfeats_"
    else:
        feat_descriptor = "unpairedfeats_"
    results_filename += feat_descriptor
    user_filename += feat_descriptor
    date = str(datetime.datetime.now())
    day = date[6:10]
    time = date[11:16]
    time = time[0:2] + "-" + time[3:5]
    results_filename += day + "_" + time + ".csv"
    user_filename += day + "_" + time + ".csv"
    return results_filename, user_filename


def load_data_matrices(data_type):
    global modeled_book_lil_matrix, modeled_book_lil_matrix_2
    if data_type == "tfidf":
        modeled_book_lil_matrix, not_used, num_terms = \
            get_list_of_modeled_book_lists_for_all_books(True)
        num_terms_secondary_matrix = None
    elif data_type == "sentiment":
        modeled_book_lil_matrix, not_used, num_terms = \
            get_list_of_modeled_book_lists_for_all_books(False)
        num_terms_secondary_matrix = None
    else:
        modeled_book_lil_matrix, not_used, num_terms = \
            get_list_of_modeled_book_lists_for_all_books(True)
        modeled_book_lil_matrix_2, not_used, num_terms_secondary_matrix = \
            get_list_of_modeled_book_lists_for_all_books(False)
    return num_terms, num_terms_secondary_matrix


def run_full_test(data_type, classifier_type):
    if "both" in data_type:
        second_matrix_has_info = True
    else:
        second_matrix_has_info = False

    print("Loading modeled books...")
    num_terms, second_num_terms = load_data_matrices(data_type)

    print("Loading user training ratings...")
    global list_of_user_rating_tuples_eval
    user_ids, list_of_user_rating_tuples_eval = \
        make_list_of_user_rating_tuples_from_file(evaluation_data_filename)
    global list_of_user_rating_tuples_training

    print("Loading user development/evaluation ratings...")
    user_ids, list_of_user_rating_tuples_training = \
        make_list_of_user_rating_tuples_from_file(training_data_filename)
    assert len(list_of_user_rating_tuples_training) == len(list_of_user_rating_tuples_eval), \
        "Different number of users in training and development/evaluation datasets"

    results_filename, user_filename = make_test_filenames(data_type, classifier_type)
    results = open(results_filename, "w")
    users = open(user_filename, "w")

    users.write("user_id,good_training_ratings,total_training_ratings,fraction_correct," +
                "total_dev_ratings\n")
    results.write("user_id,isbn,actual_rating,actual_rating_binary,prediction_log_prob_bad," +
                  "prediction_log_prob_good,prediction_prob_bad,prediction_prob_good,correct\n")

    total_users = len(list_of_user_rating_tuples_eval)
    for user_index in range(total_users):
        user_eval_ratings = list_of_user_rating_tuples_eval[user_index]
        if len(user_eval_ratings) > 0:
            if int(((user_index - 1)/total_users)*1000) != int(((user_index)/total_users)*1000):
                print(str(int(1000*user_index/total_users)) +
                      "/1000 of users done at " + str(datetime.datetime.now()))
            user_id = user_ids[user_index]
            if data_type == "both_indep":
                profile = UserProfileLearner(classifier_type, num_terms, second_num_terms, True, None,
                                             second_matrix_has_info)
            elif data_type == "tfidf":
                profile = UserProfileLearner(classifier_type, num_terms, second_num_terms, False, True,
                                             second_matrix_has_info)
            else:
                profile = UserProfileLearner(classifier_type, num_terms, second_num_terms, False, False,
                                             second_matrix_has_info)
            training_pairs = {0:[], 1:[]}
            user_training_ratings = list_of_user_rating_tuples_training[user_index]
            #max_bad, user_training_ratings_binary = \
            #    binarify_ratings_by_average(user_training_ratings)
            if not user_id.startswith("A"):
                # BXuser, so convert rating tuples to 1-5 ratings first
                for i in range(len(user_training_ratings)):
                    user_training_ratings[i] = (user_training_ratings[i][0],
                                                1 + (user_training_ratings[i][1]/2.5))
            max_bad = 2.5
            user_training_ratings_binary = binarify_ratings_with_max_bad_rating(max_bad,
                                                                                user_training_ratings)
            num_good_training_ratings = 0
            for rating in user_training_ratings_binary:
                if rating[1] == 0:
                    training_pairs[0].append(rating[0])
                else:
                    training_pairs[1].append(rating[0])
                    num_good_training_ratings += 1
            profile.train(training_pairs)

            num_correct = 0
            for rating in user_eval_ratings:
                isbn = indices_to_isbns[rating[0]]
                unnormalized_log_probs = profile.predict_rating(rating[0])
                normalized_unlog_probs, unused = unlog_log_probs(unnormalized_log_probs, True)
                if rating[1] <= max_bad:
                    binary_rating = 0
                else:
                    binary_rating = 1
                if (binary_rating == 0 and normalized_unlog_probs[0] > .5) or \
                    (binary_rating == 1 and normalized_unlog_probs[1] > .5):
                    correct = 1
                else:
                    correct = 0
                num_correct += correct
                results.write(user_id + "," + isbn + "," + str(rating[1]) + "," +
                              str(binary_rating) + "," + str(unnormalized_log_probs[0]) + "," +
                              str(unnormalized_log_probs[1]) + "," +
                              str(normalized_unlog_probs[0]) + "," +
                              str(normalized_unlog_probs[1]) + "," + str(correct) + "\n")
            users.write(user_id + "," + str(num_good_training_ratings) + "," +
                        str(len(user_training_ratings)) + "," +
                        str(num_correct / len(user_eval_ratings)) + "," +
                        str(len(user_eval_ratings)) + "\n")

    print("Test complete!")
    print("Results in " + results_filename + "\nand " + user_filename + ".")
    results.close()
    users.close()


def main():
    data_type = argv[1]
    nb_or_maxent = argv[2]
    max_num_feats = int(argv[3].strip())
    global MAX_NUM_FEATURES, MAX_NUM_FEATURE_CANDIDATES
    if (nb_or_maxent.startswith("n") and USE_PAIRED_FEATURES_NAIVE_BAYES) or \
         (nb_or_maxent.startswith("m") and USE_PAIRED_FEATURES_MAXENT):
        MAX_NUM_FEATURES = max_num_feats
        MAX_NUM_FEATURE_CANDIDATES = 2 * MAX_NUM_FEATURES
    elif (nb_or_maxent.startswith("n") and not USE_PAIRED_FEATURES_NAIVE_BAYES) or \
            (nb_or_maxent.startswith("m") and not USE_PAIRED_FEATURES_MAXENT):
        MAX_NUM_FEATURES = max_num_feats // 2
        MAX_NUM_FEATURE_CANDIDATES = max_num_feats
    else:
        print("Please enter either \"naive_bayes\" or \"maxent\" as the classifier type.")
        exit(1)
    if data_type.startswith("t") and nb_or_maxent.startswith("n"):
        run_full_test("tfidf", "naive_bayes")
    elif data_type.startswith("t") and nb_or_maxent.startswith("m"):
        run_full_test("tfidf", "maxent_package")
    elif data_type.startswith("s") and nb_or_maxent.startswith("n"):
        run_full_test("sentiment", "naive_bayes")
    elif data_type.startswith("s") and nb_or_maxent.startswith("m"):
        run_full_test("sentiment", "maxent_package")
    elif data_type.startswith("b") and "indep" in data_type and nb_or_maxent.startswith("n"):
        run_full_test("both_indep", "naive_bayes")
    elif data_type.startswith("b") and "indep" in data_type and nb_or_maxent.startswith("m"):
        run_full_test("both_indep", "maxent_package")
    elif data_type.startswith("b") and nb_or_maxent.startswith("n"):
        run_full_test("both_tfidf_determined", "naive_bayes")
    elif data_type.startswith("b") and nb_or_maxent.startswith("m"):
        run_full_test("both_tfidf_determined", "maxent_package")
    else:
        print("Could not determine which type of test to run. Quitting now.")


    #print_tfidf_vector_summary("0439136350", 20)


if __name__ == "__main__":
    main()
