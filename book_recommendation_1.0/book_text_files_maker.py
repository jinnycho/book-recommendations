"""
book_text_files_maker.py
Author: Sofia Serrano

To run this script to make the text files, give one command line argument:
the filename for the file containing the ISBNs to work on. If you wanted to
run this script on all the ISBNs, for example, you would use the following
command:
python3 book_text_files_maker.py datafile/ISBNNumsInCommon.txt

Assumes the existence of a directory named [whatever the value
of text_file_directory_name below is]; fills that directory with one
text file per ISBN we're using. Each text file will contain one line,
which will be parseable as a Python list of lists of string words,
where each inner list of words represents a sentence.

Requires following downloads:
- NLTK:
  sudo pip3 install -U nltk
- Underlying Stanford POS Tagger (only required if using TextFileMaker):
  Download from http://nlp.stanford.edu/software/tagger.shtml

and following files/directories:
- a directory named text_file_directory_name (the value of the variable below)
- datafile/GoodReadsDescriptions.txt
- datafile/GoogleBooksDescriptions.txt
- datafile/Books_5.json
    a file of JSON dictionaries (sorted by ISBNs) where each
    dictionary (which is its own line) contains the key "reviewText"
    and "asin" (where the "asin" field is the ISBN).
    We used Amazon review data obtained from the following URL:
    http://jmcauley.ucsd.edu/data/amazon/links.html
"""

import ast
import string
if __name__ == '__main__': # following is only necessary if we're using TextFileMaker
    from sys import exit
    from sys import argv
    from json import loads
    from pathlib import Path
    from re import sub
    from nltk.tag import StanfordPOSTagger
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from os import remove
    from multiprocessing import Pool
    import datetime


text_file_directory_name = "isbn_text_files"
filename_of_isbns_to_make_files_for = None # Given as command line argument
stanford_directory = "stanford-postagger-2016-10-31"
pos_training_model_to_use = "english-left3words-distsim.tagger"


def get_word_list_from_file(string_isbn):
    """
    Assuming books' text files are already created in [text_file_directory_name], this
    function returns the prepared list of words
    :param string_isbn: the string version of the isbn to get text for
    :return: a list of lists of strings, where each inner list represent a sentence
      and each string within that list represents a word
    """
    f = open(text_file_directory_name + "/" + string_isbn + ".txt", "r")
    string_list = f.readline().strip()
    f.close()
    return ast.literal_eval(string_list)


class TextFileMaker:
    # Make string containing all characters to strip from the tokens
    strip_chars = string.whitespace + string.punctuation
    misc_to_delete = ["&apos;s", "’s", "'s", "“", "”"]
    sentence_enders = ['.', '?', '!']
    amazon_data = None
    goodreads_api_data = None
    google_api_data = None
    cur_amazon_review = None
    cur_goodreads_book = None
    cur_google_book = None
    cur_gr_book_has_string = False
    cur_google_book_has_string = False
    lemmatizer = WordNetLemmatizer()
    tagger = StanfordPOSTagger(stanford_directory + "/models/" + pos_training_model_to_use,
                               path_to_jar=(stanford_directory + "/stanford-postagger-3.7.0.jar"),
                               encoding="utf8")
    try:
        amazon_data = open("datafile/Books_5.json", "r")
        goodreads_api_data = open("datafile/GoodReadsDescriptions.txt", "r")
        google_api_data = open("datafile/GoogleBooksDescriptions.txt", "r")
        cur_amazon_review = loads(amazon_data.readline())
        cur_goodreads_book = goodreads_api_data.readline()
        cur_google_book = google_api_data.readline()
        cur_gr_book_has_string = True
        cur_google_book_has_string = True
    except:
        print("Can't find one or more of the following files:")
        print("\tdatafile/Books_5.json (Amazon data)")
        print("\tdatafile/GoodReadsDescriptions.txt")
        print("\tdatafile/GoogleBooksDescriptions.txt")
        print("Quitting now.")
        try:
            amazon_data.close()
            goodreads_api_data.close()
            google_api_data.close()
        except:
            pass
        exit(1)

    def make_unlemmatized_isbn_file(self, isbn_string):
        """
        Makes the unlemmatized text file corresponding to this ISBN number.
        Will not make a file if *either* the unlemmatized or lemmatized text file for this
        ISBN already exists in the directory.
        :param isbn_string: the isbn number, in string form
        :return: True if it makes a new file, False if it finds one already there and
         doesn't replace it (makes and saves a text file)
        """
        filename = text_file_directory_name + "/" + isbn_string + "Unlemmatized.txt"
        test_file = Path(filename)
        test_file_2 = Path(text_file_directory_name + "/" + isbn_string + ".txt")
        if test_file.is_file() or test_file_2.is_file():
            return False

        api_isbn = "ISBN" + isbn_string
        list_of_sentences_about_book = []

        if self.cur_gr_book_has_string and \
                        api_isbn > self.cur_goodreads_book[0:self.cur_goodreads_book.index(':')]:
            # we've passed cur_goodreads_book and need to update it
            while self.cur_gr_book_has_string and \
                        api_isbn > self.cur_goodreads_book[0:self.cur_goodreads_book.index(':')]:
                self.get_next_api_book(True)
        if self.cur_gr_book_has_string and \
                        api_isbn == self.cur_goodreads_book[0:self.cur_goodreads_book.index(':')]:
            # we have GoodReads text for this book
            sentence_strings = self.break_text_into_sentence_list(
                                self.cur_goodreads_book[self.cur_goodreads_book.index(':') + 1:])
            for sentence in sentence_strings:
                list_to_append = self.tokenize_and_clean_sentence(sentence)
                if len(list_to_append) > 0:
                    list_of_sentences_about_book.append(list_to_append)
            self.get_next_api_book(True)

        if self.cur_google_book_has_string and \
                        api_isbn > self.cur_google_book[0:self.cur_google_book.index(':')]:
            # we've passed cur_google_book and need to update it
            while self.cur_google_book_has_string and \
                            api_isbn > self.cur_google_book[0:self.cur_google_book.index(':')]:
                self.get_next_api_book(False)
        if self.cur_google_book_has_string and \
                        api_isbn == self.cur_google_book[0:self.cur_google_book.index(':')]:
            # we have Google text for this book
            sentence_strings = self.break_text_into_sentence_list(
                                self.cur_google_book[self.cur_google_book.index(':') + 1:])
            for sentence in sentence_strings:
                list_to_append = self.tokenize_and_clean_sentence(sentence)
                if len(list_to_append) > 0:
                    list_of_sentences_about_book.append(list_to_append)
            self.get_next_api_book(False)

        if isbn_string > self.cur_amazon_review["asin"]:
            # we've passed the book cur_amazon_review is reviewing and need to update it
            while isbn_string > self.cur_amazon_review["asin"]:
                self.cur_amazon_review = loads(self.amazon_data.readline())
        while isbn_string == self.cur_amazon_review["asin"]:
            sentence_strings = \
                self.break_text_into_sentence_list(self.cur_amazon_review["reviewText"])
            for sentence in sentence_strings:
                list_to_append = self.tokenize_and_clean_sentence(sentence)
                if len(list_to_append) > 0:
                    list_of_sentences_about_book.append(list_to_append)
            self.cur_amazon_review = loads(self.amazon_data.readline())

        f = None
        try:
            f = open(filename, "w")
        except:
            print("Could not create files because directory named " + text_file_directory_name +
                  "\ndoes not exist. Please create directory and then run script again.")
            self.amazon_data.close()
            self.goodreads_api_data.close()
            self.google_api_data.close()
            exit(1)
        f.write(str(list_of_sentences_about_book) + '\n')
        f.close()
        return True

    def get_next_api_book(self, goodreads):
        """
        Updates either self.cur_goodreads_book or self.cur_google_book
        (and possibly also self.cur_gr_book_has_string and self.cur_google_book_has_string,
        if we run out of descriptions from a particular API)
        :param goodreads: if True, then update cur_goodreads_book; otherwise, update
            cur_google_book
        :return: None
        """
        updated_yet = False
        if goodreads:
            while self.cur_gr_book_has_string and not updated_yet:
                try:
                    self.cur_goodreads_book = self.goodreads_api_data.readline()
                    if self.cur_goodreads_book.strip() != "":
                        updated_yet = True
                except:
                    self.cur_gr_book_has_string = False
        else:
            while self.cur_google_book_has_string and not updated_yet:
                try:
                    self.cur_google_book = self.google_api_data.readline()
                    if self.cur_google_book.strip() != "":
                        updated_yet = True
                except:
                    self.cur_google_book_has_string = False

    def break_text_into_sentence_list(self, text):
        """
        Breaks review/description text into a list of sentences
        :param text: the text of a single review or description
        :return: a list of strings, where each string is a sentence
        """
        sentence_list = []
        chunks_to_break_up = [text]
        for punctuation_mark in self.sentence_enders:
            temp = []
            for chunk in chunks_to_break_up:
                temp += chunk.split(punctuation_mark)
            chunks_to_break_up = temp
        # now remove all empty chunks
        for possible_sentence in chunks_to_break_up:
            if possible_sentence != "":
                sentence_list.append(possible_sentence)
        return sentence_list

    def tokenize_and_clean_sentence(self, book_sentence):
        """
        Partially written by Ryan Gorey
        Returns a list version of the input sentence, where each word is lowercase and has any
        leading and/or trailing whitespace or punctuation removed
        :param book_sentence: A string containing a single sentence about a book
        :return: A list of strings, where each string is a cleaned, normalized version of a
            single word in the sentence
        """
        raw_tokens = book_sentence.split()
        stripped_cleaned_tokens = []
        for token in raw_tokens:
            for substring in self.misc_to_delete:
                token = token.replace(substring, '')
            # delete html tags
            token = sub("<[^>]*>", '', token)
            token = sub("&lt;[^(&gt;)]*&gt;", '', token)
            token = sub("&lt[^(&gt)]*&gt", '', token)
            token = sub("&[^;]*;", '', token)

            stripped_token = token.strip(self.strip_chars)
            if stripped_token != "":
                lowercase_word = ""
                for i in range(len(stripped_token)):
                    lowercase_word += stripped_token[i].lower()
                stripped_cleaned_tokens.append(lowercase_word)

        return stripped_cleaned_tokens

    def make_lemmatized_isbn_file(self, isbn_string):
        """
        Makes the lemmatized version of a particular file.
        Will not remake a file if lemmatized version is already found in directory
        :param isbn_string: a string version of the ISBN to make a file for
        :return: True if it makes a new file, False if it does not
        """
        filename = text_file_directory_name + "/" + isbn_string + ".txt"
        test_file = Path(filename)
        if test_file.is_file():
            return False
        f = open(text_file_directory_name + "/" + isbn_string + "Unlemmatized.txt", "r")
        string_list = f.readline().strip()
        f.close()
        lists_of_unlemmatized_words = ast.literal_eval(string_list)
        list_of_lists_of_tagged_words = self.get_tagged_lists_of_words(lists_of_unlemmatized_words)
        list_of_lists_of_lemmatized_words = self.lemmatize_full_list(list_of_lists_of_tagged_words)

        f = None
        try:
            f = open(filename, "w")
        except:
            print("Could not create files because directory named " + text_file_directory_name +
                  "\ndoes not exist. Please create directory and then run script again.")
            self.amazon_data.close()
            self.goodreads_api_data.close()
            self.google_api_data.close()
            exit(1)
        f.write(str(list_of_lists_of_lemmatized_words) + '\n')
        f.close()
        remove(text_file_directory_name + "/" + isbn_string + "Unlemmatized.txt")
        return True

    def lemmatize_full_list(self, tagged_full_list):
        """
        Lemmatizes several sentences' words
        :param tagged_full_list: list of lists (representing sentences) of tuples
         (representing words), where each tuple consists of a string word and its
         string Treebank POS tag
        :return: a list of lists of strings, where every string is the lemmatized
         word from the original sentence
        """
        list_to_return = []
        word_dict = {}
        num_sentences = len(tagged_full_list)
        so_far = 0
        for sentence in tagged_full_list:
            num_words = len(sentence)
            for i in range(num_words):
                orig_pos_tag = sentence[i][1]
                lemmatized_pos_tag = self.get_simplified_pos_tag(orig_pos_tag)
                if lemmatized_pos_tag == '':
                    sentence[i] = (sentence[i][0], )
                else:
                    sentence[i] = (sentence[i][0], lemmatized_pos_tag)
            lemmatized_sentence, word_dict = self.lemmatize_sentence(sentence, word_dict)
            so_far += 1
            if so_far / 1000 == so_far // 1000:
                print(str(so_far) + " lemmatized sentences out of " + str(num_sentences))
            list_to_return.append(lemmatized_sentence)
        return list_to_return

    def lemmatize_sentence(self, tupled_sentence, word_dict):
        """
        Lemmatizes a sentence and updates a word_dict with word/POS-tag tuples as keys,
        and their lemmatized form as values. (Hoping this will save time for some of the
        books that have a *lot* written about them; doesn't seem to add much, if any,
        time in practice.)
        :param tupled_sentence: a list of word/POS-tag (where POS-tags are the wordnet.POS
         objects) tuples in a sentence
        :param word_dict: dictionary as described above
        :return: lemmatized_sentence (list of strings), updated word_dict
        """
        # first, check word_dict to see how many of these we already know the answer to
        cur_dict_keys = word_dict.keys()
        indices_to_insert_words = []
        lemmatized_sentence = []
        num_words = len(tupled_sentence)
        new_tuples = []
        key_tuples = []
        for i in range(num_words):
            word_tuple = tupled_sentence[i]
            key_tuple = word_tuple
            if len(word_tuple) == 2:
                if word_tuple[1] == wordnet.ADJ:
                    key_tuple = (word_tuple[0], 'A')
                elif word_tuple[1] == wordnet.VERB:
                    key_tuple = (word_tuple[0], 'V')
                elif word_tuple[1] == wordnet.NOUN:
                    key_tuple = (word_tuple[0], 'N')
                else:
                    key_tuple = (word_tuple[0], 'R')
            if key_tuple in cur_dict_keys:
                lemmatized_sentence.append(word_dict[key_tuple])
            else:
                lemmatized_sentence.append("dummyword")
                indices_to_insert_words.append(i)
                new_tuples.append(word_tuple)
                key_tuples.append(key_tuple)
        lemmatized_new_words = self.lemmatize_multithreaded(new_tuples)
        num_new_words = len(indices_to_insert_words)
        for i in range(num_new_words):
            new_tuple = key_tuples[i]
            new_lemmatized_word = lemmatized_new_words[i]
            word_dict[new_tuple] = new_lemmatized_word
            lemmatized_sentence[indices_to_insert_words[i]] = new_lemmatized_word
        return lemmatized_sentence, word_dict

    def lemmatize_multithreaded(self, sentence_in_word_tuples, cores=1):
        with Pool(processes=cores) as pool:
            result = pool.starmap(self.lemmatizer.lemmatize, sentence_in_word_tuples)
        return result

    def get_simplified_pos_tag(self, orig_pos_tag):
        """
        Converts a Treebank tag (with more information) to its simplified
        wordnet tag equivalent, if it has one
        :param orig_pos_tag: a Treebank tag expressed as a string
        :return: the wordnet tag equivalent, or '' if no equivalent found
        """
        if orig_pos_tag.startswith('J'):
            return wordnet.ADJ
        elif orig_pos_tag.startswith('V'):
            return wordnet.VERB
        elif orig_pos_tag.startswith('N'):
            return wordnet.NOUN
        elif orig_pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def get_tagged_lists_of_words(self, list_of_lists_of_words):
        """
        Uses the NLTK interface for the Stanford Part of Speech tagger to tag each
        string in one of the given lists of words with its part of speech.
        Only makes one call to the POS tagger because of how much time each call takes,
        since calls on longer lists don't seem to take noticeably longer than calls
        on shorter lists.
        :param list_of_lists_of_words: a list of lists of strings
        :return: a list of lists of tuples, where every tuple consists of a string word
         and a string tag representing its part of speech
        """
        inds_to_break_before = [0]
        aggregated_list = []
        for sentence in list_of_lists_of_words:
            inds_to_break_before.append(len(sentence) +
                                        inds_to_break_before[len(inds_to_break_before) - 1])
            aggregated_list += sentence
        try:
            aggregated_list = self.tagger.tag(aggregated_list)
        except:
            # something failed while tagging the words all together. It'll
            # take too long to do them individually, so split up the list into
            # every 1000 words
            list_with_tag_tuples = []
            num_batches = len(aggregated_list) / 1000
            if num_batches % 1.0 == 0.0:
                num_batches = int(num_batches)
            else:
                num_batches = int(num_batches) + 1
            for i in range(num_batches):
                batch = self.tagger.tag(aggregated_list[1000 * i: (1000 * (i + 1))])
                list_with_tag_tuples += batch
            aggregated_list = list_with_tag_tuples
        for i in range(len(list_of_lists_of_words)):
            list_of_lists_of_words[i] = aggregated_list[inds_to_break_before[i]:
                                                        inds_to_break_before[i + 1]]
        return list_of_lists_of_words

    def make_all_files_not_already_there(self):
        isbns = open(filename_of_isbns_to_make_files_for, "r")
        print("Data files successfully loaded, beginning to make files at " +
              str(datetime.datetime.now()) + ":")
        counter = 0
        for isbn in isbns:
            isbn = isbn.strip()
            if isbn != "":
                new_file_made = self.make_unlemmatized_isbn_file(isbn)
                assert (not new_file_made), "File was overwritten and should not have been."
                new_file_made = self.make_lemmatized_isbn_file(isbn)
                if not new_file_made:
                    print("Lemmatized version of " + isbn + ".txt already exists. " +
                          "File not overwritten.")
                counter += 1
                if counter // 100 == counter / 100:
                    print(str(counter) + " files done in total at " + str(datetime.datetime.now()))
        print("Done at " + str(datetime.datetime.now()) + "!")
        isbns.close()
        self.amazon_data.close()
        self.goodreads_api_data.close()
        self.google_api_data.close()


def main():
    fm = TextFileMaker()

    global filename_of_isbns_to_make_files_for
    filename_of_isbns_to_make_files_for = argv[1].strip()
    fm.make_all_files_not_already_there()


if __name__ == "__main__":
    main()
