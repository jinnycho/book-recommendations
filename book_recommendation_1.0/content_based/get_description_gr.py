'''
Get Description for Books
Authors: JordiKai Watanabe-Inouye, Sofia Serrano

Uses the GoodReads API to pull book descriptions from the Good Reads Database and
adds the description as a field in our own database.

To get an API key for Goodreads, go to:
https://www.goodreads.com/api
Then replace the api_key variable below with that key.
'''

import requests
# Documentation of python pkg http://lxml.de/parsing.html
from xml.etree import ElementTree
from timed_file_writer import TimedFileWriter

# acquire api_key from https://www.goodreads.com/api
api_key = "insert_api_key_here"

'''
Arguments: An ISBN
Returns: Request URL

Takes an ISBN and builds a GET request URL
    ex)  https://www.goodreads.com/book/isbn/ISBN?format=xml&key=api_key&isbn=0002005395
'''
def build_url_request(isbn):
    url = "https://www.goodreads.com/book/isbn/" + isbn.strip() + "?format=xml&key=" + api_key
    return url

'''
Arguments: A URL
Returns: XML Response

Takes a URL and using the requests pkg sends a GET request to the GoodReads API
'''
def get_response(url):
    response = requests.get(url)
    return response

def main():
    # Open file_writer to write ISBN & description pair
    file_writer = TimedFileWriter(1, 100000000, "datafile/GoodReadsDescriptionsNew.txt")
    # Grab the ISBN Nums from the Instersection
    isbns = open("datafile/ISBNNumsInCommon.txt", "r")
    for isbn in isbns:
        if (isbn[len(isbn) - 1] is '\n'):
            isbn = isbn[0:(len(isbn) - 1)]
        url = build_url_request(isbn)
        response = get_response(url)
        # For successful API call, response code will be 200 (OK)
        if(response.ok):
            tree = ElementTree.fromstring(response.content)
            description = tree.findall("./book/description")
            if len(description) != 0:
                description = ElementTree.tostring(description[0], encoding='unicode')
                if description.strip() is not "":
                    description = description.strip()
                    if (description.startswith("<description>")):
                        description = description[13:]
                    if (description.endswith("</description>")):
                        if len(description) is not 14:
                            description = description[0:len(description)-14]
                        else:
                            description = ""
                    if description is not "" and "<description />" not in description:
                        file_writer.write_book_description_to_file(description, isbn)


main()
