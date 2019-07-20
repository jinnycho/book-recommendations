"""
max_ent_weight_trainer.py
Author: Sofia Serrano
Optimizes weights of a maximum entropy classifier with features already chosen
and weights already initialized.
Implements Improved Iterative Scaling algorithm to train the weights.
train_weights() is the function that user_profile_learner calls from here.

HAS NOT BEEN SWITCHED OVER TO WORK WITH SCIPY DATA FORMAT
"""
from math import e
from math import pow
from math import log
from math import fabs
from scipy.optimize import fsolve
from numpy import exp  # Might complain about exp or say this is unused, but don't remove--
                       # used in solving parsed string equations


def unlog_log_probs(log_probs, normalize):
    """
    Goes through three steps:
        1. Finds the minimum log prob and adds its absolute value to all of the log probs
           in the list, which effectively multiplies all of the corresponding non-log
           probabilities by the same factor (does this so that we won't lose a bunch of
           information when we unlog)
        2. Unlogs the probabilities
        3. Normalizes the non-log probabilities and returns them
    :param log_probs: the log probabilities to be unlogged (and then normalized)
    :param normalize: True if probabilities should be normalized, False if not
    :return: two things:
        1) the normalized unlogged probabilities, in a list of the same length as log_probs
        2) the (very negative) power of e by which all the probabilities were shifted up
           during calculations
    """
    # find minimum log_prob
    min_log_prob = 1000
    for log_prob in log_probs:
        if log_prob < min_log_prob:
            min_log_prob = log_prob
    probs = []
    # probs will be unlogged version of probs, shifted up by |min_log_prob| first
    for log_prob in log_probs:
        try:
            probs.append(pow(e, log_prob - min_log_prob))
        except:
            # one value was too big, so redo probs
            probs = []
            for log_prob in log_probs:
                if min_log_prob < log_prob:
                    probs.append(1)
                else:
                    probs.append(0)
            break
    if normalize:
        prob_sum = 0
        for prob in probs:
            prob_sum += prob
        normalized_probs = []
        for prob in probs:
            normalized_probs.append(prob/prob_sum)
        return normalized_probs, min_log_prob
    else:
        return probs, min_log_prob


class MaxEntWeightTrainer:
    maxent_classifier = None
    binary_features = None
    emp_p_book_rating = None
    emp_p_book = None
    power_e_to_mult_by = 0
    training_features = None
    orig_book_vectors = None
    max_weight_training_iterations = None
    stop_training_threshold = None

    def __init__(self, maxent_classifier, training_features, orig_book_vectors,
                 use_binary_features, max_weight_training_iterations, stop_training_threshold):
        self.maxent_classifier = maxent_classifier
        self.binary_features = use_binary_features
        self.training_features = training_features
        self.orig_book_vectors = orig_book_vectors
        self.max_weight_training_iterations = max_weight_training_iterations
        self.stop_training_threshold = stop_training_threshold

    def train_weights(self):
        """
        Trains the feature weights of the classifier according to the Improved Iterative
        Scaling algorithm
        :return: None (modifies self.maxent_classifier's feature_weights
        """
        self.put_together_empirical_probability_tables()
        counter = 0
        already_printed = False
        # could theoretically do this as a while loop waiting for convergence to 0...
        for i in range(self.max_weight_training_iterations):
            total_change_made = self.adjust_all_weights()
            if total_change_made < self.stop_training_threshold:
                print("Done training MaxEnt weights after " + str(counter + 1) + " iterations.")
                already_printed = True
                break
            counter += 1
        if not already_printed:
            print("Trained MaxEnt weights for " + str(counter) + " iterations.")

    def adjust_all_weights(self):
        """
        Executes one iteration of the Improved Iterative Scaling algorithm
        :return: Total absolute value of change made
        """
        abs_val_of_changes_made = 0
        num_bad_features = len(self.maxent_classifier.feature_weights[0])
        for feature_index in range(len(self.training_features[0])):
            delta = self.calculate_change_in_weight(feature_index)
            if feature_index % num_bad_features != feature_index:
                self.maxent_classifier.feature_weights[1][feature_index % num_bad_features] += delta
            else:
                self.maxent_classifier.feature_weights[0][feature_index] += delta
            abs_val_of_changes_made += fabs(delta)
        return abs_val_of_changes_made

    def calculate_change_in_weight(self, feature_index):
        """
        Calculates the amount by which one binary-valued feature's weight should change.
        :param feature_index: the index of the feature weight to change
        :return: the amount by which the feature weight should change
        """
        if self.binary_features:
            return self.calculate_change_in_weight_binary(feature_index)
        else:
            return self.calculate_change_in_weight_binary(feature_index)

    def calculate_change_in_weight_binary(self, feature_index):
        """
        Calculates the amount by which one binary-valued feature's weight should change,
        according to the equation given at the end of Adam Berger's paper.
        :param feature_index: the index of the feature weight to change
        :return: the amount by which the feature weight should change
        """
        num_bad_features = len(self.maxent_classifier.feature_weights[0])
        feature_is_bad = (feature_index < num_bad_features)

        constant = 0
        # all books have all features in the same order: first all the bad features,
        # then all the good features
        for book_ind in range(len(self.training_features)):
            book = self.training_features[book_ind]
            fi_x_y = book[feature_index]["feat"]
            emp_p_x_y = self.emp_p_book_rating[book_ind]
            constant += emp_p_x_y * fi_x_y

        coeffs = []
        exp_coeffs = []
        for book_ind in range(len(self.training_features)):
            book = self.training_features[book_ind]

            emp_p_x = self.emp_p_book[book_ind]
            probs = self.maxent_classifier.predict_rating_max_ent(self.orig_book_vectors[book_ind])
            probs, unused_info = unlog_log_probs(probs, True)
            if feature_is_bad:
                p_y_given_x = probs[0]
            else:
                p_y_given_x = probs[1]
            fi_x_y = book[feature_index]["feat"]
            coeffs.append(emp_p_x * p_y_given_x * fi_x_y)

            # now get sum of all features that apply for this rating/book combination
            feat_sum = 0
            if feature_is_bad:
                for i in range(num_bad_features):
                    feat_sum += book[i]["feat"]
            else:
                i = num_bad_features

                while i < len(book):
                    feat_sum += book[i]["feat"]
                    i += 1
            exp_coeffs.append(feat_sum)

        str_exp = str(coeffs[0]) + " * exp(" + str(exp_coeffs[0]) + "*x)"
        for i in range(len(coeffs)):
            if i == 0:
                pass
            str_exp += "+" + str(coeffs[i]) + " * exp(" + str(exp_coeffs[i]) + "*x)"
        str_exp += " - " + str(constant)

        func = eval("lambda x: " + str_exp)
        try:
            solution = fsolve(func, 0.5)
        except:
            print("Trouble finding solution to 0 = " + str_exp + ".\nReturning .000001 instead.")
            return .000001
        try:
            int(solution[0])
        except:
            print("Trouble processing solution to 0 = " + str_exp + ".\nReturning .000001 instead.")
            return .000001
        return solution[0] * pow(e, self.power_e_to_mult_by)

    def calculate_change_in_weight_nonbinary(self, feature_index):
        """
        THIS IS NOT CURRENTLY WORKING.
        In theory, though, it would calculate an incremental change for one nonbinary
        feature's feature weight.
        This implements the delta-finding equation given in the paper by Kamal Nigam,
        John Lafferty, and Andrew McCallum (I think, unless there's a mistake I haven't
        been able to find yet).
        :param feature_index: The index of the feature weight to be changed
        :return: the amount by which the corresponding weight should be changed
        """
        num_bad_features = len(self.maxent_classifier.feature_weights[0])
        feature_is_bad = (feature_index < num_bad_features)

        constant = 0
        # all books have all features in the same order: first all the bad features,
        # then all the good features
        for book_ind in range(len(self.training_features)):
            book = self.training_features[book_ind]
            fi_x_y = book[feature_index]["feat"]
            if (feature_is_bad and book[0]["rating"] == 0) or \
               ((not feature_is_bad) and book[0]["rating"] == 1):
                constant += fi_x_y

        coeffs = []
        exp_coeffs = []
        for book_ind in range(len(self.training_features)):
            book = self.training_features[book_ind]

            probs = self.maxent_classifier.predict_rating_max_ent(self.orig_book_vectors[book_ind])
            probs, unused_info = unlog_log_probs(probs, True)
            if feature_is_bad:
                p_y_given_x = probs[0]
            else:
                p_y_given_x = probs[1]
            fi_x_y = book[feature_index]["feat"]
            coeffs.append(p_y_given_x * fi_x_y)

            # now get sum of all features that apply for this rating/book combination
            feat_sum = 0
            if feature_is_bad:
                for i in range(num_bad_features):
                    feat_sum += book[i]["feat"]
            else:
                i = num_bad_features
                while i < len(book):
                    feat_sum += book[i]["feat"]
                    i += 1
            exp_coeffs.append(feat_sum)

        str_exp = str(coeffs[0]) + " * exp(" + str(exp_coeffs[0]) + "*x)"
        for i in range(len(coeffs)):
            if i == 0:
                pass
            str_exp += "+" + str(coeffs[i]) + " * exp(" + str(exp_coeffs[i]) + "*x)"
        str_exp += " - " + str(constant)

        func = eval("lambda x: " + str_exp)
        try:
            solution = fsolve(func, 0.5)
        except:
            print("Trouble finding solution to 0 = " + str_exp + ".\nReturning .000001 instead.")
            return .000001
        try:
            int(solution[0])
        except:
            print("Trouble processing solution to 0 = " + str_exp + ".\nReturning .000001 instead.")
            return .000001
        return solution[0] * pow(e, self.power_e_to_mult_by)

    def put_together_empirical_probability_tables(self):
        """
        Fills out empirical probability tables that will be used in training
        the feature weights
        :return: None (fills out self.emp_p_book and self.emp_p_book_rating)
        """
        self.emp_p_book_rating = []
        self.emp_p_book = []
        self.fill_out_empirical_prob_tables()

    def fill_out_empirical_prob_tables(self):
        """
        Fills out empirical probability tables that will be used in training
        the feature weights
        :return: None (fills out self.emp_p_book and self.emp_p_book_rating)
        """
        num_bad_books = 0
        bad_books = []
        good_books = []
        for book in self.training_features:
            if book[0]["rating"] == 0:
                num_bad_books += 1
                bad_books.append(book)
            else:
                good_books.append(book)
        total_books = len(self.training_features)
        # put together emp_p_book_and_rating
        self.emp_p_book = []
        self.emp_p_book_rating = []
        for book in bad_books:
            self.emp_p_book_rating.append(1 / total_books)
        for book in good_books:
            self.emp_p_book_rating.append(1 / total_books)
        for book in self.training_features:
            self.emp_p_book.append(1 / total_books)
