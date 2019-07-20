import json
import csv


def get_list_of_intersection_isbns():
    """ Pulls all the isbns from the file of intersection isbns into one list """
    f = open("datafile/ISBNNumsInCommon.txt", "r")
    list_to_return = []
    for line in f:
        line = line.lstrip().rstrip()
        if line != "":
            list_to_return.append(line)
    f.close()
    return list_to_return


def get_num_of_intersection_reviews():
    """ Gets the total number of text reviews for all of the 37,563 books in the Amazon dataset that we'll be using. Takes advantage of the fact that all the ISBNs in the file of intersection ISBNs *and* in the Amazon review dataset are sorted. The number it gave was 951,578 (just so we don't have to run this again) """
    isbns = get_list_of_intersection_isbns()
    f = open("reviews_Books_5.json", "r")
    total_reviews_in_intersection = 0
    for review in f:
        if review.lstrip().rstrip() == "":
            continue
        # review will now be a dictionary, not a string
        review = json.loads(review)
        if review["asin"] == isbns[0]:
            total_reviews_in_intersection += 1
        elif review["asin"] > isbns[0]:
            # if the review's isbn is past the first isbn still in the intersection list
            isbns.remove(isbns[0])
            if len(isbns) == 0:
                break
            if  review["asin"] == isbns[0]:
                total_reviews_in_intersection += 1
    f.close()
    return total_reviews_in_intersection


def get_num_of_intersection_reviewers_Amazon():
    """ Gets the total number of unique reviewers in the Amazon dataset who reviewed the intersection books. The number should be 286,189"""
    isbns = get_list_of_intersection_isbns()
    f = open("reviews_Books_5.json", "r")
    # use a set structure so that there will be no repetitions in the unique_reviewers set
    unique_reviewers = set()
    for reviewer in f:
        if reviewer.lstrip().rstrip() == "":
            continue
        # reviewer will now be a dictionary, not a string
        reviewer = json.loads(reviewer)
        if reviewer["asin"] == isbns[0]:
            unique_reviewers.add(reviewer["reviewerID"])
        elif reviewer["asin"] > isbns[0]:
            # if the review's isbn is past the first isbn still in the intersection list
            isbns.remove(isbns[0])
            if len(isbns) == 0:
                break
            if  reviewer["asin"] == isbns[0]:
                unique_reviewers.add(reviewer["reviewerID"])
    f.close()
    return len(unique_reviewers)


def get_subset_BX():
    """ Gets the subset of the BX dataset that contains only information about books in the intersection"""
    isbns = get_list_of_intersection_isbns()
    ratings = open("BX-Book-Ratings.csv", "r")
    c = csv.writer(open("BXsubset.csv", "wb"))
    for line in ratings:
        info_list = line.rstrip("\r\n").split(",")
        isbn = info_list[1]
        # since the isbn numbers in the ISBNNumsInCommon.txt file has leading 0s, we add 0s to our isbn
        while len(isbn) < 10:
                isbn = "0" + isbn
        if isbn in isbns:
            c.writerow([info_list[0], isbn, info_list[2]])
            # With the constructed BXsubset.csv file, can use R to easily compute the number of unique BX users who rated the intersection books.


def get_subset_Amazon():
    """ Gets the subset of the Amazon dataset that contains isbn, user_id and numeric rating for books in the intersection"""
    isbns = get_list_of_intersection_isbns()
    f = open("reviews_Books_5.json", "r")
    c = csv.writer(open("AmazonSubset.csv", "w"))
    c.writerow(["user_id", "isbn", "rating"]) # set names of the columns
    for reviewInfo in f:
        if reviewInfo.lstrip().rstrip() == "":
            continue
        # reviewInfo will now be a dictionary, not a string
        reviewInfo = json.loads(reviewInfo)
        if reviewInfo["asin"] == isbns[0]:
            c.writerow([reviewInfo["reviewerID"], reviewInfo["asin"], reviewInfo["overall"]])
        elif reviewInfo["asin"] > isbns[0]:
            # if the review's isbn is past the first isbn still in the intersection
            # list
            isbns.remove(isbns[0])
            if len(isbns) == 0:
                break
            if  reviewInfo["asin"] == isbns[0]:
                c.writerow([reviewInfo["reviewerID"], reviewInfo["asin"], reviewInfo["overall"]])
    f.close()



def main():
    #    print("\nTotal number of text reviews in the part of the Amazon data that\n"+ "we'll be using: " + str(get_num_of_intersection_reviews()) + "\n")
    # print("\nTotal number of unique reviewers in the part of the Amazon data that\n"+ "we'll be using: " + str(get_num_of_intersection_reviewers_Amazon()) + "\n")
    # get_subset_BX()
    get_subset_Amazon()

if __name__ == "__main__": main()
