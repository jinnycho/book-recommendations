import datetime

'''
Arguments: None
Returns: Array with AB ISBN

Takes AmazonBooks (AB) txt file and reads in the ISBN to an array.
Since input file is sorted, the array will also be sorted
'''
def read_amazon_isbns_from_file():
    isbns = open("Datafile/sortedAmazonISBNs.txt", "r")
    list_to_return = []
    for line in isbns:
        line = line.lstrip().rstrip()
        if line is not "":
            list_to_return.append(line)
    isbns.close()
    return list_to_return

'''
Arguments: None
Returns: Array with BX ISBN

Takes BookCrossing (BX) csv file and reads in the ISBN to an array.
Since input file is sorted, the array will also be sorted
'''
def read_bookx_isbns_from_file():
    # Since this is a CSV file we had to change the encoding so we could readin the file properly
    isbns = open("Datafile/BX-Books.csv", "r", encoding="ISO-8859-1")
    list_to_return = []
    for line in isbns:
        line = line.lstrip()
        comma_ind = line.find(",")
        if comma_ind >= 0 and line.lstrip()[0: comma_ind] is not "ISBN":
            isbn_num = line.lstrip()[0: comma_ind]
            while len(isbn_num) < 10:
                isbn_num = "0" + isbn_num
            list_to_return.append(isbn_num)
    isbns.close()
    return list_to_return

'''
Arguments: None
Returns: A txt file with the intsection of the two data sets.

Creates a txt file with the intersection by ISBN of both data sets
'''
def produce_file_of_isbns_in_common():
    num_in_common = 0
    write_to = open("Datafile/ISBNNumsInCommon.txt", "w")
    amazon_list = read_amazon_isbns_from_file()
    bookx_list = read_bookx_isbns_from_file()
    amazon_ind = 0
    bookx_ind = 0
    total_amazon = len(amazon_list)
    total_bookx = len(bookx_list)
    while amazon_ind < total_amazon and bookx_ind < total_bookx:
        amazon_isbn = amazon_list[amazon_ind]
        bookx_isbn = bookx_list[bookx_ind]
        if amazon_isbn == bookx_isbn:
            write_to.write(amazon_isbn + "\n")
            num_in_common += 1
            amazon_ind += 1
            bookx_ind += 1
        elif amazon_isbn < bookx_isbn:
            amazon_ind += 1
        elif bookx_isbn < amazon_isbn:
            bookx_ind += 1
    write_to.close()
    print("The two datasets have " + str(num_in_common) + " books in common.")

def main():
    produce_file_of_isbns_in_common()

if __name__ == "__main__": main()
