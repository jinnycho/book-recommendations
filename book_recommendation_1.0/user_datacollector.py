"""
JordiKai & Sofia

Preprocessing that sorts BookCrossing data.
Maps a given user U to a list of tuples [(ISBN, rating)...]
"""

import random

def collect_data():
    sorted_data_filename_1 = "datafile/AmazonSorted.csv"
    sorted_data_filename_2 = "datafile/BXSorted.csv"
    count_books_to_users = dict()
    new_file = open("datafile/user_ratings.txt", "w")

    data = open(sorted_data_filename_1, "r")
    curr_user = ""
    num_books_for_user = -1
    for line in data:
        if line.strip() == "":
            continue
        elif "user_id" in line:
            continue
        attributes = [x.strip() for x in line.split(',')]
        if "\"" in attributes[1]:
            attributes[1] = attributes[1][1:(len(attributes[1]) - 1)]
        isbn = attributes[2]
        if "\"" in isbn:
            isbn = isbn[1:len(isbn) - 1]
        while len(isbn) < 10:
            isbn = "0" + isbn
        attributes[2] = isbn

        if curr_user != attributes[1]:
            if num_books_for_user > -1:
                if num_books_for_user in count_books_to_users.keys():
                    count_books_to_users[num_books_for_user] += 1
                else:
                    count_books_to_users[num_books_for_user] = 1
            num_books_for_user = 0
            curr_user = attributes[1]
            new_file.write("\n" + curr_user + ": ")

        new_file.write("(" + attributes[2] + ", " + attributes[3] + "); ")
        num_books_for_user += 1
    if num_books_for_user in count_books_to_users.keys():
        count_books_to_users[num_books_for_user] += 1
    else:
        count_books_to_users[num_books_for_user] = 1
    data.close()
    data = open(sorted_data_filename_2, "r")
    curr_user = ""
    num_books_for_user = -1
    for line in data:
        if line.strip() == "":
            continue
        elif "user_id" in line:
            continue
        attributes = [x.strip() for x in line.split(',')]
        if "\"" in attributes[1]:
            attributes[1] = attributes[1][1:len(attributes[1]) - 1]
        isbn = attributes[2]
        if "\"" in isbn:
            isbn = isbn[1:len(isbn) - 1]
        while len(isbn) < 10:
            isbn = "0" + isbn
        attributes[2] = isbn

        if curr_user != attributes[1]:
            if num_books_for_user > -1:
                if num_books_for_user in count_books_to_users.keys():
                    count_books_to_users[num_books_for_user] += 1
                else:
                    count_books_to_users[num_books_for_user] = 1
            num_books_for_user = 0
            curr_user = attributes[1]
            new_file.write("\n" + curr_user + ": ")

        new_file.write("(" + attributes[2] + ", " + attributes[3] + "); ")
        num_books_for_user += 1
    if num_books_for_user in count_books_to_users.keys():
        count_books_to_users[num_books_for_user] += 1
    else:
        count_books_to_users[num_books_for_user] = 1
    data.close()
    new_file.close()

    new_file = open("datafile/book_counts_for_users.txt", "w")
    rating_counts = count_books_to_users.keys()
    rating_counts = sorted(rating_counts)
    for rating_count in rating_counts:
        new_file.write(str(rating_count) + ": " + str(count_books_to_users[rating_count]) + "\n")
    new_file.close()

def greater_than_2_ratings():
    data = open("datafile/user_ratings.txt", "r")
    new_file = open("datafile/user_ratings_g3.txt", "w")
    # no new line until the end of a user, so we can read by line and strip at semi-colons if > 2 then write to new file
    for line in data:
        if len(line.split(";")) - 1 > 2:
            new_file.write(line)

    data.close()
    new_file.close()

def produce_user_rating_variation_data():
    data = open("datafile/user_ratings_g3.txt", "r")
    f = open("datafile/user_rating_variation.csv", "w")
    f.write("user_id,total_ratings,total_good_ratings\n")
    for line in data:
        if line.strip() == "":
            continue
        user_id = line[0:line.index(":")]
        bx_user = (not user_id.startswith("A"))
        line = line[line.index(":") + 1:].strip()
        rating_tuples = line.split(";")[:-1]
        total_ratings = len(rating_tuples)
        good_ratings = 0
        for rating in rating_tuples:
            rating = int(rating[rating.index(",") + 2: rating.index(")")])
            if bx_user:
                rating = (rating/2.5) + 1
            if rating > 2.5:
                good_ratings += 1
        f.write(user_id + "," + str(total_ratings) + "," + str(good_ratings) + "\n")
    f.close()
    data.close()

def separate_data():
    training = open("datafile/training_ratings.txt", "w")
    development = open("datafile/development_ratings.txt", "w")
    evaluation = open("datafile/evaluation_ratings.txt", "w")
    data = open("datafile/user_ratings_g3.txt", "r")

    for line in data:
        ratings = line.split(";")
        ratings = ratings[:len(ratings) - 1]
        user_id = ratings[0][:ratings[0].index(":") + 1]
        ratings[0] = ratings[0][ratings[0].index("("):]

        two_samples = random.sample(ratings, 2)
        remaining_samples = set(ratings) - set(two_samples)
        training.write(user_id + " " + two_samples[0] + "; " + two_samples[1] + ";")
        development.write(user_id + " ")
        evaluation.write(user_id + " ")
        if len(remaining_samples) == 1:
            for item in remaining_samples:
                decider = random.random()
                if decider <= .5:
                    development.write(item + ";")
                else:
                    evaluation.write(item + ";")
        else:
            training_cutoff = .6 - (2/len(ratings))
            dev_cutoff = .2
            evaluation_cutoff = .2
            total = training_cutoff + dev_cutoff + evaluation_cutoff
            training_cutoff /= total
            dev_cutoff /= total
            evaluation_cutoff /= total
            for sample in remaining_samples:
                decider = random.random()
                if decider <= training_cutoff:
                    training.write(" " + sample + ";")
                elif decider <= training_cutoff + dev_cutoff:
                    development.write(" " + sample + ";")
                else:
                    evaluation.write(" " + sample + ";")
        training.write("\n")
        development.write("\n")
        evaluation.write("\n")


    training.close()
    development.close()
    evaluation.close()
    data.close()

def main():
    # following three method calls will randomly partition the data into the 3 sets
    if False:
        collect_data()
        greater_than_2_ratings()
        separate_data()

    # for producing the csv file used to calculate variation in users' ratings
    produce_user_rating_variation_data()


if __name__ == '__main__':
    main()
