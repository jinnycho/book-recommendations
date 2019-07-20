""" Authors: Shatian Wang, Eunjin (Jinny) Cho, and JordiKai Watanabe-Inouye
Creates a nested dictionary based on either the development or the evaluation data files.
These dictionaries will be used in kNN to determine what books to predict. They are also
used to compute the rmse and accuracy, since they contain the actual ratings """

import pickle
from collections import defaultdict
from create_util_mat import UtilMat

def file_to_dict():
    """Creates the nested dictioary user --> dict of books --> ratings
    by reading either or dev or eval txt file """
    # dev_data = open("datafile/development_ratings.txt", "r")
    dev_data = open("datafile/evaluation_ratings.txt", "r")
    # load dictionaries
    util_mat_obj = pickle.load(open("datafile/UtilMat_obj.p", "rb"))
    dev_dict = defaultdict()

    for line in dev_data:
        rating_dict = defaultdict()
        info_list = line.rstrip("\r\n").split(":")
        user = info_list[0]
        user_index = util_mat_obj.user_to_index_dict[user]
        item_list = info_list[1].replace(" ", "").split(";")[:-1]
        for item in item_list:
            isbn = item.strip("()").split(",")[0].lstrip('0')
            rating = float(item.strip("()").split(",")[1])
            book_index = util_mat_obj.isbn_to_index_dict[isbn]
            rating_dict[book_index] = rating
        dev_dict[user_index] = rating_dict
    return dev_dict


def save(dict):
    """ Save the dict obj passed in """
    # pickle.dump(dict, open("datafile/dev_dict.p", "wb"))
    pickle.dump(dict, open("datafile/eval_dict.p", "wb"))


def load():
    """ Loads and returns either the dev or eval dict"""
    # return pickle.load(open("datafile/dev_dict.p", "rb"))
    return pickle.load(open("datafile/eval_dict.p", "rb"))


def test():
    """ Test function that allows us to take a peak at the dict to ensure that the dictionary was produced correctly"""
    dev_dict = load()
    print("---- Loaded dict -----")
    util_mat_obj = pickle.load(open("datafile/UtilMat_obj.p", "rb"))
    for i in range(20):
        print(util_mat_obj.index_to_user_dict[i])
        for book_idx in dev_dict[i]:
            print(util_mat_obj.index_to_isbn_dict[book_idx])
        print("\n")

def main():
    # dev_dict = file_to_dict()
    eval_dict = file_to_dict()
    print("---- Created dict -----")
    # print("---- Start Test ----")
    # test()
    # print("---- End Test ----")
    # save(dev_dict)
    save(eval_dict)
    print("---- Saved dict -----")


if __name__ == "__main__": main()
