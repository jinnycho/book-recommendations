"""
Timed File-Writer for Book Descriptions from APIs
Authors: Sofia Serrano

Writes book descriptions to files and puts the program to sleep if necessary to ensure that
by the time any of these methods returns, we'll be allowed to make another call to the API
"""


import datetime
import time
from math import fabs

def background_print(message, backgroundOutputFile):
    f = open(backgroundOutputFile, 'a')
    f.write(message + "\n")
    f.close()

class TimedFileWriter:
    max_number_calls_per_day = None
    min_num_seconds_bet_calls = None
    total_num_calls_today = None
    next_allowed_call_time = None
    filename = ""
    background_filename = ""
    beginning_of_next_day = None
    cur_day = 1
    
    """
        Constructor arguments:
           - Minimum number of seconds between calls (int)
           - Maximum number of calls per day (int)
           - Filename (string)
        Note: either input None for both of the first two params if timing doesn't matter at all,
        or input an actual number for both of them
    """
    def __init__(self, min_num_seconds_bet_calls, max_number_calls_per_day, filename):
        self.min_num_seconds_bet_calls = min_num_seconds_bet_calls
        self.max_number_calls_per_day = max_number_calls_per_day
        self.filename = filename
        self.beginning_of_next_day = datetime.datetime.now() + datetime.timedelta(days=1)
        self.background_filename = filename[0:(len(filename) - 4)] + 'Output.txt'
        if self.min_num_seconds_bet_calls is not None:
            self.next_allowed_call_time = datetime.datetime.now()
            self.total_num_calls_today = 0

    """
        Arguments:
           - Book description (string)
           - ISBN number (string)
        Writes a book description (and the ISBN number it goes with) to file, calling the
        appropriate method for timed or untimed, and if necessary, stalls the program until
        we're next allowed to make an API call
    """
    def write_book_description_to_file(self, book_description, isbn_number):
        if self.max_number_calls_per_day is None or self.min_num_seconds_bet_calls is None:
            self.write_file_line_untimed(book_description, isbn_number)
        else:
            self.write_file_line_timed(book_description, isbn_number)

    """
        Arguments:
           - Book description (string)
           - ISBN number (string)
        DON'T CALL THIS FUNCTION, call write_book_description_to_file instead. This is a helper
        function.
        Appends the ISBN number and the description to the file of descriptions.
    """
    def write_file_line_untimed(self, book_description, isbn_number):
        if datetime.datetime.now() > self.beginning_of_next_day:
            background_print("Still running after day " + str(self.cur_day),
                             self.background_filename)
            self.cur_day += 1
            self.beginning_of_next_day = datetime.datetime.now() + datetime.timedelta(days=1)
        f = open(self.filename, "a")
        f.write("ISBN" + isbn_number + ": " + book_description + "\n\n")
        f.close()

    """
        Arguments:
           - Book description (string)
           - ISBN number (string)
        DON'T CALL THIS FUNCTION, call write_book_description_to_file instead. This is a helper
        function.
        Appends the ISBN number and the description to the file of descriptions.
        If we need to kill time until we're next allowed to call the API, puts the program to
        sleep for the appropriate amount of time.
    """
    def write_file_line_timed(self, book_description, isbn_number):
        f = open(self.filename, "a")
        f.write("ISBN" + isbn_number + ": " + book_description + "\n\n")
        f.close()
        self.total_num_calls_today += 1
        while datetime.datetime.now() < self.next_allowed_call_time:
            time.sleep(fabs((self.next_allowed_call_time - datetime.datetime.now()).microseconds / 1000000))
        self.next_allowed_call_time = datetime.datetime.now() + \
                                   datetime.timedelta(seconds=self.min_num_seconds_bet_calls)
        background_print("\tSuccessfully wrote description at " + str(datetime.datetime.now()), self.background_filename)

        # if we've done all our allowed calls per day, put program to sleep for the remaining time
        # in this day
        if self.total_num_calls_today >= self.max_number_calls_per_day:
            background_print("Still running after day " + str(self.cur_day), self.background_filename)
            while datetime.datetime.now() < self.beginning_of_next_day:
                time.sleep(fabs((self.beginning_of_next_day - datetime.datetime.now()).seconds))
                time.sleep(fabs(
                    (self.beginning_of_next_day - datetime.datetime.now()).microseconds / 1000000))
            self.total_num_calls_today = 0
            self.next_allowed_call_time = datetime.datetime.now() + \
                                          datetime.timedelta(seconds=self.min_num_seconds_bet_calls)
            self.cur_day += 1
            self.beginning_of_next_day = datetime.datetime.now() + datetime.timedelta(days=1)
