import datetime

def produce_sorted_amazon_isbns_in_file():
    data = open("reviews_Books_5.json", "r")
    write_to = open("Datafile/sortedAmazonISBNs.txt", "w")
    last_isbn = ""
    counter = 0
    lineCounter = 0
    last_printed_isbn = "0"
    checkin_after_no_problems = datetime.datetime.now() + datetime.timedelta(minutes = 15)
    for line in data:
        lineCounter += 1
        start_index = line.find("asin") + 8
        if checkin_after_no_problems < datetime.datetime.now():
            print("No problems encountered in last 15 minutes")
            checkin_after_no_problems = datetime.datetime.now() + datetime.timedelta(minutes = 15)
        if start_index != -1:
            # if this gets us the full ISBN number
            if (line[start_index - 1] == "\"" and line[start_index + 10] == "\""):
                isbn = line[start_index: start_index + 10]
                if last_isbn != "" and isbn < last_isbn:
                    print("Two ISBN numbers out of order: " + last_isbn + ", " + isbn +
                          " in lines " + counter + " and " + (counter + 1) + ".")
                    checkin_after_no_problems = datetime.datetime.now() + datetime.timedelta(minutes = 15)
                elif last_printed_isbn != isbn:
                    last_printed_isbn = isbn
                    counter += 1
                    write_to.write(isbn + "\n")
            else:
                print("Error finding ISBN number in line " + (counter + 1) + ":\n" + line + "\n")
    write_to.close()
    print(counter)

def main():
    produce_sorted_amazon_isbns_in_file()

if __name__ == "__main__": main()
