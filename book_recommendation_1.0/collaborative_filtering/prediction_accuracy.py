"""Author: JordiKai Watanabe-Inouye
Computes the accuracy for each similarity metric (normalized and not normalized).
The accuracy is stored in a dictionary where the keys are a tuple of three items (k, T/F, sim_metric)
whose values are the accuracy. This storage method facilitcates ease of conversion to csv. """

import pickle

def ratio_correct(pred_dict, dev_dict):
    """ Calculates the accuracy for the given prediction dictionary. We do NOT distinguish between
    Correct Positive and Correct Negative OR False Negative and False Positive. We are only concerned
    with the total correct and incorrect predictions.
    :param pred_dict: a nested dictionary containing the predicted ratings for user-book pairings
    :param dev_dict: a nested dictionary containing the actual ratings for user-book pairings
    :return correct_accuracy: the accuracy of a correct prediction  """
    correct_resp = 0
    incorrect_resp = 0
    total_pred = 0
    for user in range(98268):
        total_pred += len(dev_dict[user])
        for book in dev_dict[user]:
            if pred_dict[user][book] <= 2.5 and dev_dict[user][book] <= 2.5:
                correct_resp += 1
            elif pred_dict[user][book] > 2.5 and dev_dict[user][book] > 2.5:
                correct_resp += 1
            else:
                incorrect_resp += 1
    # print("Correct response = ", correct_resp/total_pred, "\n     total number of correct_resp is ", correct_resp)
    # print("Incorrect response = ", incorrect_resp/total_pred, "\n     total number of incorrect_resp is ", incorrect_resp)
    return correct_resp/total_pred


def create_accuracy_dict():
    """ Loads previously calculated predictions - combination of k's and similarity metrics.
    Create a dictionary whose keys are (k, T/F for normalization, sim_metric) and whose
    values are the accuracy of each method. This dictionary is saved and used in generating
    a csv file to be processed in R for graphs ect.  """

    # dev_dict = pickle.load(open("datafile/dev_dict.p", "rb"))
    dev_dict = pickle.load(open("datafile/eval_dict.p", "rb"))
    final_dict = dict()

    print("--- Unnormalized Pred Dict ---")
    # With NOT normalized prediction dicts
    # for k in range(2, 31):
    for k in range(30, 31):
        # print("--- k = ", k, " ---")
        for i in range(1, 4):
            if i == 1:
                # euc_filename = "dev_unnorm_pred/prediction_dict_euclidean_" + str(k) + ".p"
                euc_filename = "eval_unnorm_pred/prediction_dict_euclidean" + str(k) + ".p"
                euc_pred_dict = pickle.load(open(euc_filename, "rb"))
                accuracy = ratio_correct(euc_pred_dict, dev_dict)
                final_dict[(k, "F", "euc")] = accuracy
            if i == 2:
                # cos_filename = "dev_unnorm_pred/prediction_dict_cosine_" + str(k) + ".p"
                cos_filename = "eval_unnorm_pred/prediction_dict_cosine" + str(k) + ".p"
                cos_pred_dict = pickle.load(open(cos_filename, "rb"))
                accuracy = ratio_correct(cos_pred_dict, dev_dict)
                final_dict[(k, "F", "cos")] = accuracy
            if i == 3:
                # adj_cos_filename = "dev_unnorm_pred/prediction_dict_adj_cosine_" + str(k) + ".p"
                adj_cos_filename = "eval_unnorm_pred/prediction_dict_adj_cosine" + str(k) + ".p"
                adj_pred_dict = pickle.load(open(adj_cos_filename, "rb"))
                accuracy = ratio_correct(adj_pred_dict, dev_dict)
                final_dict[(k, "F", "adj_cos")] = accuracy
    # After review it seems as though ['datafile/prediction_dict_cosine_30.p', 0.846393556450493] has the best accuracy

    print("--- Normalized Pred Dict ---")
    # With normalized prediction dicts
    # for k in range(2, 31):
    for k in range(30, 31):
        # print("--- k = ", k, " ---")
        for i in range(1, 4):
            if i == 1:
                # euc_filename = "dev_norm_pred/prediction_dict_euclidean_" + str(k) + ".p"
                euc_filename = "eval_norm_pred/prediction_dict_euclidean" + str(k) + ".p"
                euc_pred_dict = pickle.load(open(euc_filename, "rb"))
                accuracy = ratio_correct(euc_pred_dict, dev_dict)
                final_dict[(k, "T", "euc")] = accuracy
            if i == 2:
                # cos_filename = "dev_norm_pred/prediction_dict_cosine_" + str(k) + ".p"
                cos_filename = "eval_norm_pred/prediction_dict_cosine" + str(k) + ".p"
                cos_pred_dict = pickle.load(open(cos_filename, "rb"))
                accuracy = ratio_correct(cos_pred_dict, dev_dict)
                final_dict[(k, "T", "cos")] = accuracy
            if i == 3:
                # adj_cos_filename = "dev_norm_pred/prediction_dict_adj_cosine_" + str(k) + ".p"
                adj_cos_filename = "eval_norm_pred/prediction_dict_adj_cosine" + str(k) + ".p"
                adj_pred_dict = pickle.load(open(adj_cos_filename, "rb"))
                accuracy = ratio_correct(adj_pred_dict, dev_dict)
                final_dict[(k, "T", "adj_cos")] = accuracy

    pickle.dump(final_dict, open("datafile/eval_accuracy_dict.p", "wb"))
    

def review_accuracy():
    """ In ploting the accuracy by similarity & normalization we noticed that there were two outliers:
    Cosine with unnormalized data & Euclidean with unnormalized data. This function will help us sanity
    check the values in both of these. """
    dev_dict = pickle.load(open("datafile/dev_dict.p", "rb"))

    euc_pred_dict = pickle.load(open("datafile/prediction_dict_euclidean_2.p", "rb"))
    cos_pred_dict = pickle.load(open("datafile/prediction_dict_cosine_2.p", "rb"))

    print("Euclidean with unnormalized data k = 2 \n   ", ratio_correct(euc_pred_dict, dev_dict))
    print("Cosine with unnormalized data k = 2 \n   ", ratio_correct(cos_pred_dict, dev_dict))

    print("\n--- Check Prediction Dict ---")
    euc_pred_dict1 = pickle.load(open("test_dev/prediction_dict_euclidean_2.p", "rb"))
    cos_pred_dict1 = pickle.load(open("test_dev/prediction_dict_cosine_2.p", "rb"))

    print("Euclidean with unnormalized data k = 2 \n   ", ratio_correct(euc_pred_dict1, dev_dict))
    print("Cosine with unnormalized data k = 2 \n   ", ratio_correct(cos_pred_dict1, dev_dict))

    euc_pred_dict2 = pickle.load(open("test_dev/prediction_dict_euclidean_2_t2.p", "rb"))
    cos_pred_dict2 = pickle.load(open("test_dev/prediction_dict_cosine_2_t2.p", "rb"))

    print("Euclidean with unnormalized data k = 2 \n   ", ratio_correct(euc_pred_dict2, dev_dict))
    print("Cosine with unnormalized data k = 2 \n   ", ratio_correct(cos_pred_dict2, dev_dict))

    error = {}
    error_lst = []
    for user in euc_pred_dict1:
        error[user] = []
        for book in euc_pred_dict1[user]:
            if euc_pred_dict1[user][book] != euc_pred_dict2[user][book]:
                error[user].append(book)
                error_lst.append((user, book))
    print(len(error_lst)) #52871 user book pairings are mismatches
    for i in range(20):
        print(error_lst[i])
        user = error_lst[i][0]
        book = error_lst[i][1]
        print("    ",euc_pred_dict1[user][book], euc_pred_dict2[user][book])
        print("    acutal = ", dev_dict[user][book])
    # pickle.dump(error, open("test_dev/error.p", "wb"))

    # euc_pred_dict = pickle.load(open("datafile/prediction_dict_euclidean_30.p", "rb"))
    # cos_pred_dict = pickle.load(open("datafile/prediction_dict_cosine_30.p", "rb"))
    #
    # print("Euclidean with unnormalized data k = 30 \n   ", ratio_correct(euc_pred_dict, dev_dict))
    # print("Cosine with unnormalized data k = 30 \n   ", ratio_correct(cos_pred_dict, dev_dict))

    # # Let us also review the cummulative rmse value when k = 2
    # dev_rmse_dict = pickle.load(open("datafile/dev_cumulative_rmse_dict.p", "rb"))
    # print(dev_rmse_dict[2])


def main():
    # Calculate the accuracy of the baseline
    # dev_dict = pickle.load(open("datafile/dev_dict.p", "rb"))
    # baseline = pickle.load(open("datafile/prediction_dict_avg.p", "rb"))
    # ratio_correct(baseline, dev_dict)

    create_accuracy_dict()

    # review_accuracy()





if __name__ == "__main__": main()
