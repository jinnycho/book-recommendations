import random
import pickle
import rmse
import kNN
from create_util_mat import UtilMat

# Take in a target_entries_dict that is a nested dictionary
# {user_id:{b1:r1(actual_rating),b2:r2(actual_rating),..}}
def get_random_predictions(target_entries_dict):
    # has the same structure as target_entries_dict, but the ratings are predicted ones
    prediction_dict = {}
    for user in target_entries_dict:
        prediction_dict[user] = {}
        books_to_rate_dict = target_entries_dict[user]
        for book in books_to_rate_dict:
            random_prediction = (random.random() * 4) + 1 # so the prediction will be from 1-5
            prediction_dict[user][book] = random_prediction
    return prediction_dict

def main():
    util_mat_obj = pickle.load(open("datafile/UtilMat_obj.p", "rb"))
    training_matrix = util_mat_obj.new_matrix
    user_index_to_id_dict = util_mat_obj.index_to_user_dict
    index_to_isbn_dict = util_mat_obj.index_to_isbn_dict
    print ("-------- loaded matrix--------")

    # dev_dict = pickle.load(open("datafile/dev_dict.p", "rb"))
    dev_dict = pickle.load(open("datafile/eval_dict.p", "rb"))

    # Track nonzero entries
    nonzero_training_indices = kNN.track_rated(training_matrix)

    prediction_dict_random = get_random_predictions(dev_dict)
    # pickle.dump(prediction_dict_random, open("random_pred/dev_prediction_dict_random.p", "wb"))
    pickle.dump(prediction_dict_random, open("random_pred/eval_prediction_dict_random.p", "wb"))

    rmse_single_bin_random = rmse.get_rmse_single_bin(dev_dict, prediction_dict_random, nonzero_training_indices, user_index_to_id_dict)
    # pickle.dump(rmse_single_bin_random, open("random_pred/dev_rmse_single_bin_random.p", "wb"))
    pickle.dump(rmse_single_bin_random, open("random_pred/eval_rmse_single_bin_random.p", "wb"))

    single_bin_rmse_random, cumulative_rmse_random = rmse.get_rmse(rmse_single_bin_random)
    # pickle.dump(single_bin_rmse_random, open("random_pred/dev_single_bin_rmse_random.p", "wb"))
    pickle.dump(single_bin_rmse_random, open("random_pred/eval_single_bin_rmse_random.p", "wb"))

    print("cumulative rmse: ", cumulative_rmse_random)
    # pickle.dump(cumulative_rmse_random, open("random_pred/dev_cumulative_rmse_random.p", "wb"))
    pickle.dump(cumulative_rmse_random, open("random_pred/eval_cumulative_rmse_random.p", "wb"))

    print("-------- calculated rmse --------")

if __name__ == "__main__": main()
