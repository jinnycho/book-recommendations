"""
isbn_separator.py
Author: Sofia Serrano

Written to partition the ISBN numbers in our dataset into four (roughly)
equally sized groups

Now also includes a function to create the reversed versions of those files
"""


def make_backwards_isbns_from_4_files():
    for i in range(4):
        if i == 0:
            f = open("datafile/ISBNNumsInCommon1of4.txt", "r")
            bf = open("datafile/ISBNNumsInCommon1of4Backwards.txt", "w")
        elif i == 1:
            f = open("datafile/ISBNNumsInCommon2of4.txt", "r")
            bf = open("datafile/ISBNNumsInCommon2of4Backwards.txt", "w")
        elif i == 2:
            f = open("datafile/ISBNNumsInCommon3of4.txt", "r")
            bf = open("datafile/ISBNNumsInCommon3of4Backwards.txt", "w")
        else:
            f = open("datafile/ISBNNumsInCommon4of4.txt", "r")
            bf = open("datafile/ISBNNumsInCommon4of4Backwards.txt", "w")

        list_of_isbns = []
        for line in f:
            if line.strip() != "":
                list_of_isbns.append(line.strip() + "\n")
        list_of_isbns.reverse()
        for isbn in list_of_isbns:
            bf.write(isbn)

        f.close()
        bf.close()


def separate_all_isbns_into_4():
    num_isbns_per_new_file = 37563 / 4
    f = open("datafile/ISBNNumsInCommon.txt", "r")
    f1 = open("datafile/ISBNNumsInCommon1of4.txt", "w")
    f2 = open("datafile/ISBNNumsInCommon2of4.txt", "w")
    f3 = open("datafile/ISBNNumsInCommon3of4.txt", "w")
    f4 = open("datafile/ISBNNumsInCommon4of4.txt", "w")

    so_far = 0
    for isbn in f:
        isbn = isbn.strip()
        if isbn != "":
            if so_far < num_isbns_per_new_file:
                f1.write(isbn + "\n")
            elif so_far < 2 * num_isbns_per_new_file:
                f2.write(isbn + "\n")
            elif so_far < 3 * num_isbns_per_new_file:
                f3.write(isbn + "\n")
            else:
                f4.write(isbn + "\n")
            so_far += 1

    f.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()


def main():
    #separate_all_isbns_into_4()
    make_backwards_isbns_from_4_files()


main()