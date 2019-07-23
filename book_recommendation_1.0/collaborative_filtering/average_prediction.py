""" Author: JordiKai Watanabe-Inouye
We establish a baseline for CF's kNN by recommending a user's average for books. And calculate the rmse and accuracy given this baseline.
Ideally we'd like to better than this. However, we also observe the lack of variance with each user's ratings Since our results weren't much
better (if not worse) than predicting average. """

import pickle
import kNN
import rmse

def get_average_predictions(target_entries_dict, user_avg_rating_lst):
    """ Returns a predition dictionary whose ratings are based on a user's average rating.
    :param target_entries_dict: Either the development or evaluation dictionary that
    determines which books need to be rated
            {user_id:{b1:r1(actual_rating),b2:r2(actual_rating),..}}
    :param user_avg_rating_lst: A list that stores a user's average - to look up use the user's
    idx because it's equiv to pos in lst """
    prediction_dict = {}
    for user in target_entries_dict:
        prediction_dict[user] = {}
        books_to_rate_dict = target_entries_dict[user]
        for book in books_to_rate_dict:
            prediction_dict[user][book] = user_avg_rating_lst[user]
    return prediction_dict

def main():
    util_mat_obj = pickle.load(open("datafile/UtilMat_obj.p", "rb"))
    training_matrix = util_mat_obj.new_matrix
    user_index_to_id_dict = util_mat_obj.index_to_user_dict
    index_to_isbn_dict = util_mat_obj.index_to_isbn_dict

    print ("-------- loaded matrix--------")

    dev_dict = pickle.load(open("datafile/dev_dict.p", "rb"))

    # Track nonzero entries
    nonzero_training_indices = kNN.track_rated(training_matrix)
    user_avg_rating_lst = pickle.load(open("datafile/user_avg_rating_lst.p", "rb"))

    prediction_dict_avg = get_average_predictions(dev_dict, user_avg_rating_lst)
    pickle.dump(prediction_dict_avg, open("datafile/prediction_dict_avg.p", "wb"))

    rmse_single_bin_avg = rmse.get_rmse_single_bin(dev_dict, prediction_dict_avg, nonzero_training_indices, user_index_to_id_dict)
    pickle.dump(rmse_single_bin_avg, open("datafile/rmse_single_bin_average.p", "wb"))

    single_bin_rmse_avg, cumulative_rmse_avg = rmse.get_rmse(rmse_single_bin_avg)
    pickle.dump(single_bin_rmse_avg, open("datafile/single_bin_rmse_average.p", "wb"))

    print("cumulative rmse: ", cumulative_rmse_avg)
    print("-------- calculated rmse --------")

if __name__ == "__main__": main()
