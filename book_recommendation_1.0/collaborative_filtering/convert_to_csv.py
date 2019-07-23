"""Authors: Eunjin (Jinny) Cho and JordiKai Watanabe-Inouye
This script generates a csv file for Shatian Wang to process in R. The data written will allow us to generate
graphs for our paper and presentation. Below are the graphs we intend to generate:
    - Using the rmse and k to compare the six different combinations between sim_metric and data normalization status (dev_data)
    - Using the accuracy and k to compare the six different combinations between sim_metric and data normalization status (dev_data)
    - Using a bar graph compare the rmse values for the "best point" in dev/ eval
        NOTE that the "best point" is determined by dev and would be some combination i.e. k =30 with adjusted cosine sim & not normalized
    - Using a bar graph compare the "rmse bins" for the best point in dev/eval
        NOTE the bins represent the number of books given by training
"""

import pickle

def create_csvfile_for_graph1():
    """ Loads the relevant files and creates a csv file with the following columns:
        - k: the k number of neighbors considered when predicting ratings
        - sim_metric: the similarity metric used i.e. euclidean distance, cosine similarity, adjusted cosine similarity
        - normalized: a boolean that states whether or not the trainging data was normalized or not
        - eval/dev: a string that denotes which partition the predicted data is associated with
        - rmse : the root mean square error of the predictions (rating scale 1 to 5)
        - accuracy : the binary accuracy of the predictions
    """

    results = open("datafile/csvfile_for_graph1.csv", "w")
    results.write("k, sim_metric, normalized, eval/dev, rmse, accuracy\n")

    accuracy_dict = pickle.load(open("datafile/accuracy_dict.p", "rb"))
    dev_rmse_dict = pickle.load(open("dev_unnorm_pred_rmse/dev_unnorm_cumulative_rmse_dict.p", "rb"))
    dev_rmse_dict_norm = pickle.load(open("dev_norm_pred_rmse/dev_norm_cumulative_rmse_dict.p", "rb"))

    for key in accuracy_dict:
        accuracy = accuracy_dict[key]
        # rmse = 0.0
        if key[1] == "T":
            if key[2] == "euc":
                rmse= dev_rmse_dict_norm[key[0]][0]
            if key[2] == "cos":
                rmse = dev_rmse_dict_norm[key[0]][1]
            if key[2] == "adj_cos":
                rmse = dev_rmse_dict_norm[key[0]][2]
            results.write(str(key[0]) + "," + str(key[2]) + "," +
                    str(key[1]) + "," + "dev" + "," +
                    str(rmse) + "," + str(accuracy) + "\n")
        else:
            if key[2] == "euc":
                rmse = dev_rmse_dict[key[0]][0]
            if key[2] == "cos":
                rmse = dev_rmse_dict[key[0]][1]
            if key[2] == "adj_cos":
                rmse = dev_rmse_dict[key[0]][2]
            results.write(str(key[0]) + "," + str(key[2]) + "," +
                    str(key[1]) + "," + "dev" + "," +
                    str(rmse) + "," + str(accuracy) + "\n")

    results.close()


def create_csvfile_for_graph2():
    dev_cumulative_rmse = pickle.load(open("dev_norm_pred_rmse/cumulative_rmse_dict.p", "rb"))
    eval_cumulative_rmse = pickle.load(open("eval_norm_pred_rmse/cumulative_rmse_dict.p", "rb"))

    # TODO: determine which k cumulative rmse val to retrieve and which sim metric to retrieve [euc, cos, adj]
    dev_rmse = dev_cumulative_rmse[30][1]
    eval_rmse = eval_cumulative_rmse[30][1]

    results = open("datafile/csvfile_for_graph2.csv", "w")

    results.write("eval/dev, rmse\n")
    results.write("dev", dev_rmse)
    results.write("eval", eval_rmse)

    results.close()


def create_csvfile_for_graph3():
    """ Loads the relevant files and creates a csv file with the following columns:
        - bin_size, which indicates the number of ratings provided from training
        - dev/eval, which indicates which dataset the prediction is from
        - rmse, which is related to the rmse value of a particular bin
    Recall that the information here is still on a 1 to 5 scale, rather than a binary scale (good v. bad book) """
    results = open("datafile/csvfile_for_graph3.csv", "w")
    results.write("bin_size, eval/dev, rmse\n")

    # TODO: determine which dev & eval rmse bin files to open, these files should be the ones for the best pt
    dev_rmse_bin = pickle.load(open("dev_norm_pred_rmse/single_bin_rmse_cos_30.p", "rb"))
    eval_rmse_bin = pickle.load(open("eval_norm_pred_rmse/single_bin_rmse_cos_30.p", "rb"))

    for bin_size in dev_rmse_bin:
        dev_rmse = dev_rmse_bin[bin_size]
        results.write(str(bin_size), "dev", str(dev_rmse))
        eval_rmse = eval_rmse_bin[bin_size]
        results.write(str(bin_size), "eval", str(eval_rmse))

    results.close()


def main():
    create_csvfile_for_graph1()
    # create_csvfile_for_graph2()
    # create_csvfile_for_graph3()


if __name__ == "__main__": main()
