'''
Get Description for Books 
Authors: JordiKai Watanabe-Inouye, Ryan Gorey, Sofia Serrano

Uses the Google Books API to pull book descriptions from the Google Books Database and
adds the description as a field in our own database.

Reads in the first 1000 (or fewer) ISBNs from
datafile/ISBNNumsInCommonRewritable.txt (and removes them from the beginning of that
file)

To get an API key for Google Books, go to:
https://developers.google.com/books/docs/v1/using#APIKey
Then replace the api_key variable below with that key.
'''

# Download packages
import requests
import json
import datetime
from timed_file_writer import TimedFileWriter

# acquire api_key from https://developers.google.com/books/docs/v1/using#APIKey
api_key = "insert_string_api_key_here"

'''
Arguments: An ISBN
Returns: Request URL

Takes an ISBN and builds a GET request URL
'''
def build_url_request(isbn):
    url_front = "https://www.googleapis.com/books/v1/volumes?"
    url_end = "&key=" + api_key
    url_isbn = "q=isbn%3A" + isbn.strip()
    url = url_front + url_isbn + url_end
    return url

'''
Arguments: A URL
Returns: JSON Response

Takes a URL and using the requests pkg sends a GET request to the GoogleBooks API
'''
def get_response(url):
    try:
        response = requests.get(url)
        return response
    except:
        return None

def main():
    # Grab the ISBN Nums from the intersection (use google's file, which will be shortened
    # every day)
    isbn_file = open("datafile/ISBNNumsInCommonRewritable.txt", "r")
    isbns = []
    for isbn in isbn_file:
        if isbn.strip() != "":
            isbns.append(isbn.strip())
    isbn_file.close()
    file_tag = str(datetime.datetime.now())
    file_tag = file_tag[0:file_tag.index(' ')]
    timed_file_writer = TimedFileWriter(1, 1005, "datafile/GoogleBooksDescriptions" +
        file_tag + ".txt")
    for i in range(1000):
        if len(isbns) > 0:
            isbn = isbns[0]
            isbns.remove(isbn)
        else:
            print("Completely done!")
            break
        url = build_url_request(isbn)
        response = get_response(url)
        if response is None:
            continue
        # For successful API call, response code will be 200 (OK)
        if(response.ok):
            '''
            Loading the response data into a dict variable
            json.loads takes in only binary or string variables so using content to fetch binary content
            Loads (Load String) takes a Json file and converts into python data structure
            (dict or list, depending on JSON)
            We found that for our purposes, we need to use response.text()
            '''
            book_data = json.loads(response.text)
            # ensures that we should have an item section to extract data from
            if book_data["totalItems"] != 0:
                try:
                    # the description in nested within the JSON response
                    book_description = book_data["items"][0]["volumeInfo"]["description"]
                    if book_description.strip() is not "":
                        timed_file_writer.write_book_description_to_file(book_description,
                                                                       isbn.strip())
                except KeyError:
                    continue

    isbn_file = open("datafile/ISBNNumsInCommonRewritable.txt", "w")
    for isbn in isbns:
        isbn_file.write(isbn + "\n")
    isbn_file.close()


main()
