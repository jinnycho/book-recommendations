"""
max_ent_feature_collector.py
Author: Sofia Serrano
Decides which words from the training dataset will be the features used by the
MaxEnt classifier
entry point: user_profile_learner calls collect_feature_inds_and_training_feats
"""
from math import fabs
import ast


PRINT_TRAINING_FEATURES = False
with open("../datafile/tfidf_ind_to_sent_ind_dict.txt", "r") as file:
    tf_ind_to_sent_ind = ast.literal_eval(file.readline())


def mean_of_list(list_of_nums):
    mean = 0
    for num in list_of_nums:
        mean += num
    return mean / len(list_of_nums)


class FeatureCollector:
    binary_features = None
    max_num_features = None
    max_num_feature_candidates = None
    scale_nonbinary_features = None
    orig_book_vectors = None
    training_features = None
    feature_cutoffs = None
    feature_indices = None
    feats_always_same_for_bad_and_good = None
    num_terms = None
    num_terms_2 = None
    modeled_book_lil_matrix = None
    modeled_book_lil_matrix_2 = None
    diff_feature_full_indices = None
    diff_feature_full_indices_2 = None
    same_feature_full_indices = None
    same_feature_full_indices_2 = None
    use_both_types_vects = False
    select_indep = False
    second_matrix_has_info = False

    def __init__(self, binary_features, max_num_feats, max_num_feat_cands, scale_nonbinary_feats,
                 feats_always_same_for_bad_and_good, num_terms, num_terms_second_matrix,
                 modeled_book_lil_matrix, other_mat, select_indep, second_matrix_has_info):
        self.binary_features = binary_features
        self.max_num_features = max_num_feats
        self.max_num_feature_candidates = max_num_feat_cands
        self.scale_nonbinary_features = scale_nonbinary_feats
        self.feats_always_same_for_bad_and_good = feats_always_same_for_bad_and_good
        self.num_terms = num_terms
        self.num_terms_2 = num_terms_second_matrix
        self.modeled_book_lil_matrix = modeled_book_lil_matrix
        self.diff_feature_full_indices = [[i for i in range(self.num_terms)],
                                          [i for i in range(self.num_terms)]]
        self.same_feature_full_indices = [i for i in range(self.num_terms)]
        self.second_matrix_has_info = second_matrix_has_info
        if second_matrix_has_info:
            self.diff_feature_full_indices_2 = [[i for i in range(self.num_terms_2)],
                                              [i for i in range(self.num_terms_2)]]
            self.same_feature_full_indices_2 = [i for i in range(self.num_terms_2)]
        self.modeled_book_lil_matrix_2 = other_mat
        if second_matrix_has_info:
            self.use_both_types_vects = True
        self.select_indep = select_indep

    def collect_feature_inds_and_training_feats(self, training_pairs):
        """
        self.training_features will be filled out as a list of books. Each book's list
        will be a list of feature dicts, where each feature dict looks like the following:
            {"rating" : numerical rating given to book, "feat_type" : 0 if good feat and 1 if bad,
             "feat" : this feature's value for this particular book}
        Assumes that self.feature_indices is filled out.
        :param training_pairs: dictionary of ratings to lists of book vectors
        :return: training_features, orig_book_vectors, feature_cutoffs, feature_indices
        """
        self.determine_feature_indices(training_pairs)
        self.orig_book_vectors = []
        if self.binary_features:
            self.collect_binary_features(training_pairs)
        else:
            self.collect_nonbinary_features(training_pairs)
        if PRINT_TRAINING_FEATURES:
            self.print_training_features()
        else:
            pass
            #print("Features have been selected.")
        return self.training_features, self.orig_book_vectors, self.feature_cutoffs, \
               self.feature_indices

    def collect_binary_features(self, training_pairs):
        """
        Puts together self.feature_cutoffs and self.training_features; assumes
        self.feature_indices has already been filled out.
        First, cutoffs are determined (and saved in self.feature_cutoffs) for each chosen
        feature word in each class; then, those cutoffs are used to convert the training
        features in training_pairs into their binary versions, and those are then saved
        in self.training_features
        Structure of self.feature_cutoffs after function:
            self.feature_cutoffs will be a list of two lists, the first for "bad" features,
            and the second for "good" features.
            Inside either rating's list will be a series of lists, one for each feature.
            Each feature-list contains two things: the numerical cutoff, which is the first
            item, and a boolean describing whether that cutoff is the maximum value a word
            count needs to be to have that feature return 1, or the minimum.
        :param training_pairs: dictionary of ratings to lists of book vectors
        :return: None (results saved in self.feature_cutoffs and self.training_features)
        """
        self.training_features = []
        bad_book_inds = training_pairs[0]
        good_book_inds = training_pairs[1]
        self.feature_cutoffs = []
        # each cutoff will be a pair: 0 if the number given is the max, 1 if it's the min
        bad_cutoffs = []
        good_cutoffs = []
        if not self.feats_always_same_for_bad_and_good:
            # if the words being used as features are NOT necessarily the same in both
            for i in range(len(self.feature_indices[0])):
                feature_ind = self.feature_indices[0][i][1]
                matrix = self.feature_indices[0][i][0]
                bad_vals = []
                good_vals = []
                for book_ind in bad_book_inds:
                    if matrix == 0:
                        bad_vals.append(self.modeled_book_lil_matrix[book_ind, feature_ind])
                    else:
                        bad_vals.append(self.modeled_book_lil_matrix_2[book_ind, feature_ind])
                for book_ind in good_book_inds:
                    if matrix == 0:
                        good_vals.append(self.modeled_book_lil_matrix[book_ind, feature_ind])
                    else:
                        good_vals.append(self.modeled_book_lil_matrix_2[book_ind, feature_ind])
                cutoff, is_max = self.generate_single_cutoff(bad_vals, good_vals, False)
                bad_cutoffs.append([cutoff, is_max])
            for i in range(len(self.feature_indices[1])):
                feature_ind = self.feature_indices[1][i][1]
                matrix = self.feature_indices[1][i][0]
                bad_vals = []
                good_vals = []
                for book_ind in bad_book_inds:
                    if matrix == 0:
                        bad_vals.append(self.modeled_book_lil_matrix[book_ind, feature_ind])
                    else:
                        bad_vals.append(self.modeled_book_lil_matrix_2[book_ind, feature_ind])
                for book_ind in good_book_inds:
                    if matrix == 0:
                        good_vals.append(self.modeled_book_lil_matrix[book_ind, feature_ind])
                    else:
                        good_vals.append(self.modeled_book_lil_matrix_2[book_ind, feature_ind])
                cutoff, is_max = self.generate_single_cutoff(bad_vals, good_vals, True)
                good_cutoffs.append([cutoff, is_max])
        else:
            for i in range(len(self.feature_indices)):
                feature_ind = self.feature_indices[i][1]
                matrix = self.feature_indices[i][0]
                bad_vals = []
                good_vals = []
                for book_ind in bad_book_inds:
                    if matrix == 0:
                        bad_vals.append(self.modeled_book_lil_matrix[book_ind, feature_ind])
                    else:
                        bad_vals.append(self.modeled_book_lil_matrix_2[book_ind, feature_ind])
                for book_ind in good_book_inds:
                    if matrix == 0:
                        good_vals.append(self.modeled_book_lil_matrix[book_ind, feature_ind])
                    else:
                        good_vals.append(self.modeled_book_lil_matrix_2[book_ind, feature_ind])
                cutoff, is_max = self.generate_single_cutoff(bad_vals, good_vals, False)
                bad_cutoffs.append([cutoff, is_max])
                good_cutoffs.append([cutoff, not is_max])
        self.feature_cutoffs.append(bad_cutoffs)
        self.feature_cutoffs.append(good_cutoffs)

        num_bad_feats = len(self.feature_cutoffs[0])
        # now make the condensed versions of the book vectors
        for book_ind in bad_book_inds:
            self.orig_book_vectors.append(book_ind)
            book_vector = []
            for i in range(len(bad_cutoffs)):
                cutoff = bad_cutoffs[i]
                if not self.feats_always_same_for_bad_and_good:
                    feature_ind = self.feature_indices[0][i][1]
                    matrix = self.feature_indices[0][i][0]
                else:
                    feature_ind = self.feature_indices[i][1]
                    matrix = self.feature_indices[i][0]
                if matrix == 0:
                    val = self.modeled_book_lil_matrix[book_ind, feature_ind]
                else:
                    val = self.modeled_book_lil_matrix_2[book_ind, feature_ind]
                if ((val <= cutoff[0]) and cutoff[1]) or ((val > cutoff[0]) and not cutoff[1]):
                    book_vector.append(1)
                else:
                    book_vector.append(0)
            for i in range(len(good_cutoffs)):
                cutoff = good_cutoffs[i]
                if not self.feats_always_same_for_bad_and_good:
                    feature_ind = self.feature_indices[1][i][1]
                    matrix = self.feature_indices[1][i][0]
                else:
                    feature_ind = self.feature_indices[i][1]
                    matrix = self.feature_indices[i][0]
                if matrix == 0:
                    val = self.modeled_book_lil_matrix[book_ind, feature_ind]
                else:
                    val = self.modeled_book_lil_matrix_2[book_ind, feature_ind]
                if ((val <= cutoff[0]) and cutoff[1]) or ((val > cutoff[0]) and not cutoff[1]):
                    book_vector.append(1)
                else:
                    book_vector.append(0)
            book_vector = [0, num_bad_feats, book_vector]
            self.training_features.append(book_vector)
        for book_ind in good_book_inds:
            self.orig_book_vectors.append(book_ind)
            book_vector = []
            for i in range(len(bad_cutoffs)):
                cutoff = bad_cutoffs[i]
                if not self.feats_always_same_for_bad_and_good:
                    feature_ind = self.feature_indices[0][i][1]
                    matrix = self.feature_indices[0][i][0]
                else:
                    feature_ind = self.feature_indices[i][1]
                    matrix = self.feature_indices[i][0]
                if matrix == 0:
                    val = self.modeled_book_lil_matrix[book_ind, feature_ind]
                else:
                    val = self.modeled_book_lil_matrix_2[book_ind, feature_ind]
                if ((val <= cutoff[0]) and cutoff[1]) or ((val > cutoff[0]) and not cutoff[1]):
                    book_vector.append(1)
                else:
                    book_vector.append(0)
            for i in range(len(good_cutoffs)):
                cutoff = good_cutoffs[i]
                if not self.feats_always_same_for_bad_and_good:
                    feature_ind = self.feature_indices[1][i][1]
                    matrix = self.feature_indices[1][i][0]
                else:
                    feature_ind = self.feature_indices[i][1]
                    matrix = self.feature_indices[i][0]
                if matrix == 0:
                    val = self.modeled_book_lil_matrix[book_ind, feature_ind]
                else:
                    val = self.modeled_book_lil_matrix_2[book_ind, feature_ind]
                if ((val <= cutoff[0]) and cutoff[1]) or ((val > cutoff[0]) and not cutoff[1]):
                    book_vector.append(1)
                else:
                    book_vector.append(0)
            book_vector = [1, num_bad_feats, book_vector]
            self.training_features.append(book_vector)

    def generate_single_cutoff(self, vals_for_bad, vals_for_good, for_good):
        """
        A helper function for converting chosen features into binary. At the moment,
        the way this works is by calculating the average of a feature's value in badly-rated
        books and the average of a feature's value in well-rated books, then averaging those
        two values together to produce a cutoff which will determine whether the feature
        indicator function should produce 0 or 1.
        :param vals_for_bad: a list of all training values in bad books for this feature
        :param vals_for_good: a list of all training values in good books for this feature
        :param for_good:
        :return: 2 things:
            - the cutoff value
            - either False or True (False if the value given is the min, True if it's the max)
        """
        bad_mean = mean_of_list(vals_for_bad)
        good_mean = mean_of_list(vals_for_good)
        midpoint = (bad_mean + good_mean) / 2
        if (good_mean >= bad_mean and for_good) or (good_mean <= bad_mean and not for_good):
            return midpoint, False
        else:
            return midpoint, True

    def collect_nonbinary_features(self, training_pairs):
        """
        Keep in mind: nonbinary features with the MaxEnt classifier currently don't work.
        That said, this function fills out self.training_features, assuming that
        self.feature_indices has already been filled out.
        :param training_pairs: dictionary of ratings to lists of book vectors
        :return: None (results saved in self.training_features)
        """
        self.training_features = []
        book_totals = []
        bad_books = training_pairs[0]
        good_books = training_pairs[1]
        if type(self.feature_indices[0]) is list:
            # if the words being used as features are NOT necessarily the same in both
            for book_ind in bad_books:
                self.orig_book_vectors.append(book_ind)  # This doesn't need to be fixed for N.B.
                total = 0
                book_vector = []
                book_vector.append(0)
                book_vector.append(len(self.feature_indices[0]))
                book_vector.append([])
                for feature_ind in self.feature_indices[0]:
                    if feature_ind[0] == 0:
                        amount = self.modeled_book_lil_matrix[book_ind, feature_ind[1]]
                    else:
                        amount = self.modeled_book_lil_matrix_2[book_ind, feature_ind[1]]
                    book_vector[2].append(amount)
                    total += amount
                for feature_ind in self.feature_indices[1]:
                    if feature_ind[0] == 0:
                        amount = self.modeled_book_lil_matrix[book_ind, feature_ind[1]]
                    else:
                        amount = self.modeled_book_lil_matrix_2[book_ind, feature_ind[1]]
                    book_vector[2].append(amount)
                    if feature_ind not in self.feature_indices[0]:
                        total += amount
                self.training_features.append(book_vector)
                book_totals.append(total)
            for book_ind in good_books:
                self.orig_book_vectors.append(book_ind)   # This doesn't need to be fixed for N.B.
                total = 0
                book_vector = []
                book_vector.append(1)
                book_vector.append(len(self.feature_indices[0]))
                book_vector.append([])
                for feature_ind in self.feature_indices[0]:
                    if feature_ind[0] == 0:
                        amount = self.modeled_book_lil_matrix[book_ind, feature_ind[1]]
                    else:
                        amount = self.modeled_book_lil_matrix_2[book_ind, feature_ind[1]]
                    book_vector[2].append(amount)
                    total += amount
                for feature_ind in self.feature_indices[1]:
                    if feature_ind[0] == 0:
                        amount = self.modeled_book_lil_matrix[book_ind, feature_ind[1]]
                    else:
                        amount = self.modeled_book_lil_matrix_2[book_ind, feature_ind[1]]
                    book_vector[2].append(amount)
                    if feature_ind not in self.feature_indices[0]:
                        total += amount
                self.training_features.append(book_vector)
                book_totals.append(total)
        else:
            # if the same words in the same order are pulled out as features for both
            for book_ind in bad_books:
                self.orig_book_vectors.append(book_ind)
                total = 0
                book_vector = []
                book_vector.append(0)
                book_vector.append(len(self.feature_indices))
                book_vector.append([])
                for feature_ind in self.feature_indices:
                    if feature_ind[0] == 0:
                        amount = self.modeled_book_lil_matrix[book_ind, feature_ind[1]]
                    else:
                        amount = self.modeled_book_lil_matrix_2[book_ind, feature_ind[1]]
                    book_vector[2].append(amount)
                for feature_ind in self.feature_indices:
                    if feature_ind[0] == 0:
                        amount = self.modeled_book_lil_matrix[book_ind, feature_ind[1]]
                    else:
                        amount = self.modeled_book_lil_matrix_2[book_ind, feature_ind[1]]
                    book_vector[2].append(amount)
                    total += amount
                self.training_features.append(book_vector)
                book_totals.append(total)
            for book_ind in good_books:
                self.orig_book_vectors.append(book_ind)
                total = 0
                book_vector = []
                book_vector.append(1)
                book_vector.append(len(self.feature_indices))
                book_vector.append([])
                for feature_ind in self.feature_indices:
                    if feature_ind[0] == 0:
                        amount = self.modeled_book_lil_matrix[book_ind, feature_ind[1]]
                    else:
                        amount = self.modeled_book_lil_matrix_2[book_ind, feature_ind[1]]
                    book_vector[2].append(amount)
                for feature_ind in self.feature_indices:
                    if feature_ind[0] == 0:
                        amount = self.modeled_book_lil_matrix[book_ind, feature_ind[1]]
                    else:
                        amount = self.modeled_book_lil_matrix_2[book_ind, feature_ind[1]]
                    book_vector[2].append(amount)
                    total += amount
                self.training_features.append(book_vector)
                book_totals.append(total)
        if self.scale_nonbinary_features:
            for i in range(len(self.training_features)):
                total = book_totals[i]
                feat_list = self.training_features[i][2]
                if total != 0:
                    for i in range(len(feat_list)):
                        feat_list[i] /= total

    def determine_feature_indices(self, training_pairs):
        """
        Determine which words should be used as features in the classifier
        :param training_pairs: dictionary of ratings to lists of book vectors
        :return:
        """
        if not self.second_matrix_has_info:
            if not self.feats_always_same_for_bad_and_good:
                self.determine_diff_feature_indices_for_good_and_bad_single_mat(training_pairs, 0)
            else:
                self.determine_pairs_of_feature_indices_single_mat(training_pairs, 0)
        elif self.select_indep:
            if not self.feats_always_same_for_bad_and_good:
                self.determine_diff_feature_indices_for_good_and_bad_mixed_indep(training_pairs)
            else:
                self.determine_pairs_of_feature_indices_mixed_indep(training_pairs)
        else:
            if not self.feats_always_same_for_bad_and_good:
                self.determine_diff_feature_indices_for_good_and_bad_mixed_by_tfidf(training_pairs)
            else:
                self.determine_pairs_of_feature_indices_mixed_by_tfidf(training_pairs)

    def determine_diff_feature_indices_for_good_and_bad_mixed_indep(self, training_pairs):
        self.max_num_features = self.max_num_features // 2
        tfidf_contributed_inds = self.determine_diff_feature_indices_for_good_and_bad_single_mat(training_pairs, 0)
        sentiment_contributed_inds = self.determine_diff_feature_indices_for_good_and_bad_single_mat(training_pairs, 1)
        self.feature_indices = [tfidf_contributed_inds[0] + sentiment_contributed_inds[0],
                                tfidf_contributed_inds[1] + sentiment_contributed_inds[1]]

    def determine_pairs_of_feature_indices_mixed_indep(self, training_pairs):
        self.max_num_features = self.max_num_features // 2
        tfidf_contributed_inds = self.determine_pairs_of_feature_indices_single_mat(training_pairs, 0)
        sentiment_contributed_inds = self.determine_pairs_of_feature_indices_single_mat(training_pairs, 1)
        self.feature_indices = tfidf_contributed_inds + sentiment_contributed_inds

    def determine_diff_feature_indices_for_good_and_bad_mixed_by_tfidf(self, training_pairs):
        tfidf_inds = self.determine_diff_feature_indices_for_good_and_bad_single_mat(training_pairs, 0)
        ideal_num_bad_feats = len(tfidf_inds[0])
        ideal_num_good_feats = len(tfidf_inds[1])
        new_bad_feats = []
        i = 0
        while len(new_bad_feats) < ideal_num_bad_feats or \
                (len(new_bad_feats) < self.max_num_features and i < len(tfidf_inds[0])):
            new_feat = tfidf_inds[0][i]
            i += 1
            new_bad_feats.append(new_feat)
            corr_sent_ind = tf_ind_to_sent_ind[new_feat[1]]
            if corr_sent_ind != -1:
                new_bad_feats.append((1, corr_sent_ind))
        new_good_feats = []
        i = 0
        while len(new_good_feats) < ideal_num_good_feats or \
                (len(new_good_feats) < self.max_num_features and i < len(tfidf_inds[1])):
            new_feat = tfidf_inds[1][i]
            i += 1
            new_good_feats.append(new_feat)
            corr_sent_ind = tf_ind_to_sent_ind[new_feat[1]]
            if corr_sent_ind != -1:
                new_good_feats.append((1, corr_sent_ind))
        self.feature_indices = [new_bad_feats, new_good_feats]

    def determine_pairs_of_feature_indices_mixed_by_tfidf(self, training_pairs):
        tfidf_inds = self.determine_pairs_of_feature_indices_single_mat(training_pairs, 0)
        ideal_num_feats = len(tfidf_inds)
        new_feats = []
        i = 0
        while len(new_feats) < ideal_num_feats or (len(new_feats) < self.max_num_features and i < len(tfidf_inds)):
            new_feat = tfidf_inds[i]
            i += 1
            new_feats.append(new_feat)
            corr_sent_ind = tf_ind_to_sent_ind[new_feat[1]]
            if corr_sent_ind != -1:
                new_feats.append((1, corr_sent_ind))
        self.feature_indices = new_feats

    def determine_diff_feature_indices_for_good_and_bad_single_mat(self, training_pairs, matrix_num):
        """
        Selects MAX_NUM_FEATURES indices of words to be used as good features and
         MAX_NUM_FEATURES indices of words to be used as bad features, then saves them in
         self.feature_indices, with the same words not necessarily being selected for both
         the good and bad books.
        Currently does this by first picking the MAX_NUM_FEATURE_CANDIDATES words that
         have the most frequency overall, then by picking the first MAX_NUM_FEATURES words within
         that subset for both good and bad books.
        :param training_pairs: dict of numerical ratings mapped to lists of all vectors that
            received that rating
        :param matrix_num: 0 if self.modeled_book_lil_matrix, 1 if self.modeled_book_lil_matrix_2
        :return: self.feature_indices (in case it will be changed later)
        """
        if matrix_num == 0:
            if self.num_terms <= self.max_num_features:
                return self.diff_feature_full_indices
        else:
            if self.num_terms_2 <= self.max_num_features:
                return self.diff_feature_full_indices_2
        keys = training_pairs.keys()
        neg_freq_counts_good = []
        neg_freq_counts_bad = []
        for key in keys:
            vect_indices = training_pairs[key]
            if neg_freq_counts_bad == [] and key == 0:
                if matrix_num == 0:
                    for i in range(self.num_terms):
                        neg_freq_counts_bad.append(0)
                else:
                    for i in range(self.num_terms_2):
                        neg_freq_counts_bad.append(0)
            if neg_freq_counts_good == [] and key == 1:
                if matrix_num == 0:
                    for i in range(self.num_terms):
                        neg_freq_counts_good.append(0)
                else:
                    for i in range(self.num_terms_2):
                        neg_freq_counts_good.append(0)

            if key == 0:
                for vect_ind in vect_indices:
                    if matrix_num == 0:
                        for i in range(self.num_terms):
                            neg_freq_counts_bad[i] -= fabs(self.modeled_book_lil_matrix[vect_ind, i])
                    else:
                        for i in range(self.num_terms_2):
                            neg_freq_counts_bad[i] -= fabs(self.modeled_book_lil_matrix_2[vect_ind, i])
            elif key == 1:
                for vect_ind in vect_indices:
                    if matrix_num == 0:
                        for i in range(self.num_terms):
                            neg_freq_counts_good[i] -= fabs(self.modeled_book_lil_matrix[vect_ind, i])
                    else:
                        for i in range(self.num_terms_2):
                            neg_freq_counts_good[i] -= fabs(self.modeled_book_lil_matrix_2[vect_ind, i])
        list_of_indices_good = []
        list_of_indices_bad = []
        if matrix_num == 0:
            for i in range(self.num_terms):
                list_of_indices_good.append(i)
                list_of_indices_bad.append(i)
        else:
            for i in range(self.num_terms_2):
                list_of_indices_good.append(i)
                list_of_indices_bad.append(i)
        sorted_list_g = sorted(list_of_indices_good, key=lambda index: neg_freq_counts_good[index])
        sorted_list_b = sorted(list_of_indices_bad, key=lambda index: neg_freq_counts_bad[index])

        temp_feature_inds = []
        temp_feature_inds.append(sorted_list_b[0: self.max_num_features])
        temp_feature_inds.append(sorted_list_g[0: self.max_num_features])

        self.feature_indices = []
        self.feature_indices.append([(matrix_num, x) for x in temp_feature_inds[0]])
        self.feature_indices.append([(matrix_num, x) for x in temp_feature_inds[1]])
        return self.feature_indices

    def determine_pairs_of_feature_indices_single_mat(self, training_pairs, matrix_num):
        """
        Selects MAX_NUM_FEATURES indices of words to be used as features and saves them in
         self.feature_indices, making sure that the same words are selected for both
         the good and bad books.
        Currently does this by first picking the MAX_NUM_FEATURE_CANDIDATES words that
         have the most frequency overall, then by picking the MAX_NUM_FEATURES words within
         that subset that show the most difference between good and bad ratings.
        :param training_pairs: dict of numerical ratings mapped to lists of all vectors that
            received that rating
        :param matrix_num: 0 if self.modeled_book_lil_matrix, 1 if self.modeled_book_lil_matrix_2
        :return: self.feature_indices (in case it will be changed later)
        """
        if matrix_num == 0:
            if self.num_terms <= self.max_num_features:
                return self.same_feature_full_indices
        else:
            if self.num_terms_2 <= self.max_num_features:
                return self.same_feature_full_indices_2
        sorted_indices = self.get_indices_sorted_by_total_magnitude(training_pairs, matrix_num)
        # now resort this shortened list by the difference between the average values for
        # good and bad books

        good_totals = [0 for i in range(len(sorted_indices))]
        bad_totals = [0 for i in range(len(sorted_indices))]
        num_good = [0 for i in range(len(sorted_indices))]
        num_bad = [0 for i in range(len(sorted_indices))]
        bad_vect_inds = training_pairs[0]
        good_vect_inds = training_pairs[1]
        for vect_ind in bad_vect_inds:
            for i in range(len(sorted_indices)):
                num_bad[i] += 1
                if matrix_num == 0:
                    bad_totals[i] += self.modeled_book_lil_matrix[vect_ind, sorted_indices[i]]
                else:
                    bad_totals[i] += self.modeled_book_lil_matrix_2[vect_ind, sorted_indices[i]]
        for vect_ind in good_vect_inds:
            for i in range(len(sorted_indices)):
                num_good[i] += 1
                if matrix_num == 0:
                    good_totals[i] += self.modeled_book_lil_matrix[vect_ind, sorted_indices[i]]
                else:
                    good_totals[i] += self.modeled_book_lil_matrix_2[vect_ind, sorted_indices[i]]
        for i in range(len(good_totals)):
            if num_good[i] != 0:
                good_totals[i] /= num_good[i]
        for i in range(len(bad_totals)):
            if num_bad[i] != 0:
                bad_totals[i] /= num_bad[i]
        diff_for_words = [(good_totals[i] - bad_totals[i]) for i in range(len(good_totals))]

        tuples_to_sort = sorted(zip(diff_for_words, sorted_indices))
        resorted_indices = [y for (x, y) in tuples_to_sort]

        temp_inds = resorted_indices[0: self.max_num_features]
        self.feature_indices = [(matrix_num, x) for x in temp_inds]
        return self.feature_indices

    def get_indices_sorted_by_total_magnitude(self, training_pairs, matrix_num):
        """
        Gets a sorted list of the first NUM_FEATURE_CANDIDATES indices representing the words
        with the highest absolute-value sums across all vectors for both good and bad books
        (sorted from highest sum to lowest sum)
        :param training_pairs: dict of numerical ratings mapped to lists of all vectors that
            received that rating
        :param matrix_num: 0 if self.modeled_book_lil_matrix, 1 if self.modeled_book_lil_matrix_2
        :return: the sorted list of word indices
        """
        keys = training_pairs.keys()
        neg_freq_counts = []
        for key in keys:
            vect_inds = training_pairs[key]
            if neg_freq_counts == [] and vect_inds != []:
                if matrix_num == 0:
                    for i in range(self.num_terms):
                        neg_freq_counts.append(0)
                else:
                    for i in range(self.num_terms_2):
                        neg_freq_counts.append(0)
            for vect in vect_inds:
                if matrix_num == 0:
                    for i in range(self.num_terms):
                        neg_freq_counts[i] -= fabs(self.modeled_book_lil_matrix[vect, i])
                else:
                    for i in range(self.num_terms_2):
                        neg_freq_counts[i] -= fabs(self.modeled_book_lil_matrix_2[vect, i])
        list_of_indices = []
        if matrix_num == 0:
            for i in range(self.num_terms):
                list_of_indices.append(i)
        else:
            for i in range(self.num_terms_2):
                list_of_indices.append(i)
        sorted_list = sorted(list_of_indices, key=lambda index: neg_freq_counts[index])
        if len(sorted_list) > self.max_num_feature_candidates:
            return sorted_list[0: self.max_num_feature_candidates]
        else:
            return sorted_list

    def print_training_features(self):
        """
        Prints self.training_features in a more informative way than just calling
        print() on self.training_features
        :return: None
        """
        print("=============================== TRAINING FEATURES ===============================")
        print("Books that were rated badly:")
        prev_book_rated_badly = True
        no_good_ratings_so_far = True
        for book in self.training_features:
            # if it's a badly rated book
            if book[0]["rating"] == 0:
                assert no_good_ratings_so_far, "Badly rated book comes after well-rated book"
                print("[\t[ ", end="")
                feature_index = 0
                while book[feature_index]["feat_type"] == 0:
                    feature = book[feature_index]
                    assert feature["rating"] == 0, "Positive rating mixed into badly-rated book"
                    print(str(feature["feat"]) + ", ", end="")
                    feature_index += 1
                print("]\n\t[ ", end="")
                while feature_index < len(book):
                    feature = book[feature_index]
                    assert feature["rating"] == 0, "Positive rating mixed into badly-rated book"
                    assert feature["feat_type"] == 1, "Features out of order in a training " + \
                                                      "book:\n"+str(self.training_features)
                    print(str(feature["feat"]) + ", ", end="")
                    feature_index += 1
                print("]\t]\n")
            else:
                no_good_ratings_so_far = False
                if prev_book_rated_badly:
                    prev_book_rated_badly = False
                    print("======================================================" + \
                          "===========================")
                    print("Books that were rated well:")
                print("[\t[ ", end="")
                feature_index = 0
                while book[feature_index]["feat_type"] == 0:
                    feature = book[feature_index]
                    assert feature["rating"] == 1, "Negative rating mixed into well-rated book"
                    print(str(feature["feat"]) + ", ", end="")
                    feature_index += 1
                print("]\n\t[ ", end="")
                while feature_index < len(book):
                    feature = book[feature_index]
                    assert feature["rating"] == 1, "Negative rating mixed into well-rated book"
                    assert feature["feat_type"] == 1, "Features out of order in a training book"
                    print(str(feature["feat"]) + ", ", end="")
                    feature_index += 1
                print("]\t]\n")
        print("=================================================================================")
