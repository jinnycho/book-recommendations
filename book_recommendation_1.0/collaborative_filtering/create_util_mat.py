""" Authors: Shatian Wang, Eunjin (Jinny) Cho, and JordiKai Watanabe-Inouye
Loads BX and Amazon ratings into a utility matrix"""

import numpy as np
import pickle
from collections import defaultdict
from scipy.sparse import csr_matrix
import intersection_explorer

class UtilMat:
    def __init__(self):
        self.isbn_to_index_dict = defaultdict()
        self.index_to_isbn_dict = defaultdict()
        self.index_to_user_dict = defaultdict()
        self.user_to_index_dict = defaultdict()
        self.new_matrix = None


    def create_item_dict(self):
        '''
        Creates a dictionary for rows in matrix
        @return a two dictionaries mapping (ISBNs to index) and (index to ISBNs)
        '''

        item_index = 0
        f = open("datafile/ISBNIntersection.txt", "r")
        for isbn in f:
            self.isbn_to_index_dict[isbn.strip('\n')] = item_index
            self.index_to_isbn_dict[item_index] = isbn.strip('\n')
            item_index += 1


    def get_users(self):
        f = open("datafile/training_ratings.txt", "r")
        unique_reviewers = set()
        for row_b in f:
            info_list = row_b.rstrip("\r\n").split(":")
            user = info_list[0]
            unique_reviewers.add(user)
        f.close()
        return unique_reviewers


    def create_user_dict(self):
        '''
        Handling users (columns)
        - We create two dictionaries because of the fact that csr_matrix
        do not take large integers and it only takes integers
            1. self.user_to_index_dict = {user_ID: index}
            2. self.index_to_user_dict = {index: user_ID}
        '''

        # Add unique Amazon users to index_to_user_dict
        users = self.get_users()
        user_index = 0
        for u in users:
            self.user_to_index_dict[u] = user_index
            self.index_to_user_dict[user_index] = u
            user_index += 1


    def construct_csr_matrix(self):
        '''
        Constructs a utility matrix with rows as items and columns as users.
        In each cell is the corresponding rating a user has given to an item.
        The end matrix is 37563 by 324019.
        @return utilityMatrix
        '''
        self.create_item_dict() # creates both index_to_isbn_dict, isbn_to_index_dict
        self.create_user_dict() # creates both index_to_user_dict, user_to_index_dict
        training_data = open("datafile/training_ratings.txt", "r")

        row = []
        col = []
        data = []
        for row_b in training_data:
            info_list = row_b.rstrip("\r\n").split(":")
            user = info_list[0]
            user_index = self.user_to_index_dict[user]
            item_list = info_list[1].replace(" ", "").split(";")[:-1]
            for item in item_list:
                isbn = item.strip("()").split(",")[0].lstrip('0')
                rating = float(item.strip("()").split(",")[1])
                book_index = self.isbn_to_index_dict[isbn]
                row.append(book_index)
                col.append(user_index)
                if user[0] != 'A':
                    """ BX ratings run from 0 to 10, whereas AB ratings run from 1 to 5.
                    If a user is from BX then we need to change the rating to be on a scale of 1 to 5.
                    This preserves the fact that 0 indicates unrated books by a user."""
                    rating = (rating/2.5)+1
                data.append(rating)

        row_arr = np.asarray(row)
        col_arr = np.asarray(col)
        data_arr = np.asarray(data)

        self.new_matrix = csr_matrix((data_arr, (row_arr, col_arr)), shape=(37563, 98268))


def save(UtilMat_obj):
    pickle.dump(UtilMat_obj, open("datafile/UtilMat_obj.p", "wb"))


def main():
    mat_class = UtilMat()
    mat_class.construct_csr_matrix()
    #print(mat_class.new_matrix)
    save(mat_class)

if __name__ == "__main__": main()
