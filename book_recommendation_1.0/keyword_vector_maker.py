"""
Term Frequency - Inverse Document Frequency Matrix Constructor

Author: Ryan Gorey, Sofia Serrano

Version 0.2

* * * * * * * * * *

Input: Opens a csv-type file that contains a text-based book review
and a unique identifier for the reviewed book (ISBN). This program
anticipates one review per row in the csv file.

Output: A TF-IDF matrix with TF-IDF weights for each event in the
matrix.

General process: This program parses each book review to tokenize and
linguistically process each review. From the processed reviews, this
program then uses each unique token as a term in the TF-IDF. Tokenized
reviews for the each book are concatenated into one string, and act as
the documents for the TF-IDF. Then, TF and IDF weights are calculated.
Lastly, the TF-IDF weights are calculated, and the resulting matrix
with TF-IDF weights are shared with the user.

Version Notes:

V 0.1: Working prototype created.
V 0.2: Optimizations made for building the keywords.
v 0.2.1: Changed to work off of preprocessed documents.
v 0.2.2: Changed to save copies of work done along the way.

Required downloads:
    scipy.sparse
    numpy
and files:
    datafile/narrowed_dict.txt
    datafile/ISBNNumsInCommon.txt
    isbn_text_files (a directory full of lemmatized text files in list form)
"""

### Load Packages ###

import math
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import ast
import numpy as np
from book_text_files_maker import get_word_list_from_file

## Global Variables ##

termFileName = "datafile/narrowed_dict.txt"
term_dict = None
num_books = 0
f = open("../datafile/ISBNNumsInCommon.txt", "r")
for line in f:
    if line.strip() != "":
        num_books += 1
f.close()
num_terms = 0
f = open("../datafile/narrowed_dict.txt", "r")
for line in f:
    if line.strip() != "":
        num_terms += 1
f.close()

### Methods ###

'''
Output: a "document" for a book from a file.
'''
def getBookDoc(rawLists):
    doc = []
    ### SOFIA'S CODE TO CONVERT TEXT FILES TO DOCS GOES HERE ###
    return doc

'''
Output: the list of terms (which should be in alphabetical order)
to be used in the TFIDF Matrix.
'''
def loadTermsList(termFileName):
    global term_dict
    term_dict = {}
    termFile = open(termFileName, "r")
    counter = 0
    for line in termFile:
        if line.strip() != "":
            term_dict[line.strip()] = counter
            counter += 1

'''
Output: the columns of the TFIDF identified by ISBN in a text file

:TFIDFMatrix: - a TFIDF matrix (numpy)
'''
def saveTFIDFMatrix(TFIDFMatrix):
    TFIDFMatrix = csr_matrix(TFIDFMatrix)
    np.savez("datafile/TFIDF.npz", data = TFIDFMatrix.data, indices = TFIDFMatrix.indices,
             indptr = TFIDFMatrix.indptr, shape = TFIDFMatrix.shape)
    print("TFIDF matrix successfully saved!")

'''
Fill in later
'''
def load_TFIDF_Matrix():
    loader = np.load("datafile/TFIDF.npz")
    matrix = lil_matrix(csr_matrix((loader['data'], loader['indices'], loader['indptr'] ), shape = loader['shape']))
    print("TF-IDF matrix successfully loaded from file.")
    return matrix

'''
Output: A matrix filled in with TF scores
'''
def calculateTFScores():
    TFMatrix = lil_matrix((num_books, num_terms))

    isbns = []
    isbnf = open("datafile/ISBNNumsInCommon.txt", "r")
    for isbn in isbnf:
        if isbn.strip() != "":
            isbns.append(isbn.strip())
    isbnf.close()

    term_totals = [0 for i in range(num_books)]

    for i in range(len(isbns)):
        isbn = isbns[i]
        if isbn.strip() == "":
            continue

        sentences_list = get_word_list_from_file(isbn.strip())

        for sentence in sentences_list:
            for word in sentence:
                if word in term_dict.keys():
                    index = term_dict[word]
                else:
                    index = -1
                if index != -1:
                    TFMatrix[i, index] += 1
                    term_totals[i] += 1
        if i % 100 == 0:
            print(str(i) + " text documents processed")

    cx = coo_matrix(TFMatrix)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        TFMatrix[i, j] = v / term_totals[i]

    print("TF scores done being calculated")

    return TFMatrix

'''
Output: A vector of IDF scores, with the first value corresponding to
the IDF score for the first row of TF matrix, the second value
corresponding to the IDF score for the second row of the TF matrix,
et. cetera. Also makes a backup file with one IDF score per row.

:numpyMatrix: - a numpy matrix with normalized TF scores.
'''
def calculateIDFScores(TFMatrix):
    IDFVector = [0 for i in range(num_terms)]

    cx = coo_matrix(TFMatrix)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if v != 0:
            IDFVector[j] += 1

    print("Finished term document counts for IDF calculation")

    for j in range(len(IDFVector)):
        num_docs_with_term_count = IDFVector[j]
        IDFVector[j] = 1 + math.log(float(num_books) / (num_docs_with_term_count + 1))

    print("Finished IDF vector")

    return IDFVector

'''
Output: A TFIDF Matrix, which also records a backup of the intermediate
processes in a text file.

:TFMatrix: - a numpy matrix that has the TF scores used to calculate the
TFIDF Scores.

:IDFVector: - a list that has the IDF scores for each row of the matrix.
'''
def calculateTFIDFScores(TFMatrix, IDFVector):

    cx = coo_matrix(TFMatrix)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        TFMatrix[i, j] = v * IDFVector[j]
    TFIDFMatrix = TFMatrix

    print("Finished making TF-IDF matrix")

    return TFIDFMatrix

'''
Output: The TFIDF Matrix.

:terms: The terms list used for the TFIDF.
'''
def makeTFIDFMatrix():
    TFMatrix = calculateTFScores()
    IDFVector = calculateIDFScores(TFMatrix)
    TFIDFMatrix = calculateTFIDFScores(TFMatrix, IDFVector)
    return TFIDFMatrix

def main():
    loadTermsList(termFileName)
    TFIDFMatrix = makeTFIDFMatrix()
    saveTFIDFMatrix(TFIDFMatrix)


main()
