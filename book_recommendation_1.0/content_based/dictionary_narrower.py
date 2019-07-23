"""
dictionary_narrower.py
Authors: Sofia Serrano and Ryan Gorey

Prepares all dictionaries required for book vector generation.

Assumes that isbn_text_files (or the value of text_file_directory_name in book_text_files_maker.py)
is full of preprocessed text files for each book.

Requires following downloads:
- NLTK:
  sudo pip3 install -U nltk
- Underlying Stanford POS Tagger (only required if using TextFileMaker):
  Download from http://nlp.stanford.edu/software/tagger.shtml

Requires following files:
    datafile/ISBNNumsInCommon.txt
    isbn_text_files (a directory full of lemmatized text files in list form)
    datafile/SentDict.csv
        a CSV file with words in the first field, and numbers
        representing their (non-neutral) sentiment values in the second
        field
    datafile/stop_words.txt
        a text file with one stop word per line
    datafile/modifiers_hinges_plain.txt (included in repo)
        a text file with one modifier or hinge word per line
    datafile/modifiers_vals.txt
        includes modifiers' values, as well as any words coming before
        or after them that would make them not a modifier (and a 1 if
        that extra word would be coming before, or a 2 if after)
"""

from sys import argv
import ast
from book_text_files_maker import get_word_list_from_file
from nltk.tag import StanfordPOSTagger

threshold = 10 # there should be at least [threshold] occurrences of word across all text files
               # to be included in dictionary

stanford_directory = "stanford-postagger-2016-10-31"
pos_training_model_to_use = "english-left3words-distsim.tagger"
isbn_file = "datafile/ISBNNumsInCommon.txt"

class DictionaryNarrower:
    sorted_word_list = []
    count_list = []
    tagger = StanfordPOSTagger(stanford_directory + "/models/" + pos_training_model_to_use,
                               path_to_jar=(stanford_directory + "/stanford-postagger-3.7.0.jar"),
                               encoding="utf8")

    def __init__(self, dict_name, stop_file):
        self.load_dict_minus_stop_words(dict_name, stop_file)

    def load_dict_minus_stop_words(self, dict_name, stop_file):
        stopf = open(stop_file, "r")
        stop_words = []
        for word in stopf:
            if word.strip() != "":
                stop_words.append(word.strip())
        stopf.close()
        stop_words = sorted(stop_words)

        dictf = open(dict_name, "r")
        for word in dictf:
            word = word.strip()
            if word.strip() != "" and word not in stop_words:
                self.sorted_word_list.append(word)
                self.count_list.append(0)
        dictf.close()
        print("Finished loading initial dictionary minus stop words.")

    def tally_words_from_all_text(self):
        isbns = open(isbn_file, "r")
        counter = 0
        for isbn in isbns:
            isbn = isbn.strip()
            if isbn != "":
                sentence_list = get_word_list_from_file(isbn)
                for sentence in sentence_list:
                    for word in sentence:
                        self.increment_capped_word_count_for(word)
            counter += 1
            if counter / 1000 == counter // 1000:
                print(str(counter) + " ISBNs' words cross-referenced so far")
        isbns.close()

    def increment_capped_word_count_for(self, word):
        cur_lower_bound = 0
        cur_upper_bound = len(self.sorted_word_list) - 1
        decider_index = cur_upper_bound // 2
        up_or_down = DictionaryNarrower.compare(word, self.sorted_word_list[decider_index])
        while up_or_down != 0:
            if cur_upper_bound == cur_lower_bound:
                break
            if up_or_down == 1:
                # look at words further along alphabetically
                if cur_upper_bound == decider_index + 1:
                    decider_index += 1
                    cur_lower_bound = decider_index
                else:
                    cur_lower_bound = decider_index
                    decider_index = (cur_lower_bound + cur_upper_bound) // 2
            else:
                # look at words alphabetically earlier
                if cur_lower_bound == decider_index - 1:
                    decider_index -= 1
                    cur_upper_bound = decider_index
                else:
                    cur_upper_bound = decider_index
                    decider_index = (cur_lower_bound + cur_upper_bound) // 2
            up_or_down = DictionaryNarrower.compare(word, self.sorted_word_list[decider_index])
        if self.sorted_word_list[decider_index] == word and \
              self.count_list[decider_index] < threshold:
            self.count_list[decider_index] += 1

    @staticmethod
    def compare(word, dict_word):
        if word < dict_word:
            return -1
        elif word > dict_word:
            return 1
        else:
            return 0

    def write_all_words_above_threshold_to_file(self, filename):
        f = open(filename, "w")
        num_candidates = len(self.sorted_word_list)
        for i in range(num_candidates):
            if self.count_list[i] >= threshold:
                f.write(self.sorted_word_list[i] + "\n")
        f.close()

    @staticmethod
    def clean_noun_file():
        noun_file = open("datafile/nouns.txt", "r")
        noun_list = []
        for noun in noun_file:
            noun = noun.strip()
            if noun != "":
                noun_list.append(noun)
        noun_file.close()

        opinion_words = {}
        op_file = open("datafile/SentDict.csv", "r")
        for line in op_file:
            line = line.strip()
            if line != "":
                word_and_val = line.split(",")
                word_and_val[0] = word_and_val[0].lower()
                opinion_words[word_and_val[0]] = int(word_and_val[1])
            while word_and_val[0] in noun_list:
                noun_list.remove(word_and_val[0])
        op_file.close()

        mods_hinges_file = open("datafile/modifiers_hinges_plain.txt", "r")
        for word in mods_hinges_file:
            word = word.strip()
            if word != "":
                while word in noun_list:
                    noun_list.remove(word)
                if word in opinion_words.keys():
                    opinion_words.pop(word, None)
        mods_hinges_file.close()

        mods_hinges_file = open("datafile/stop_words.txt", "r")
        for word in mods_hinges_file:
            word = word.strip()
            if word != "":
                while word in noun_list:
                    noun_list.remove(word)
                if word in opinion_words.keys():
                    opinion_words.pop(word, None)
        mods_hinges_file.close()

        noun_file = open("datafile/fewer_nouns.txt", "w")
        for noun in noun_list:
            noun_file.write(noun + "\n")
        noun_file.close()

        op_file = open("datafile/op_dict.txt", "w")
        op_file.write(str(opinion_words))
        op_file.close()

    @staticmethod
    def python_readable_files():
        mod_dict = {}
        mod_file = open("datafile/modifiers_vals.txt", "r")
        for line in mod_file:
            line = line.strip()
            if line != "":
                word_val_caveats = line.split(",")
                word_val_caveats[1] = float(word_val_caveats[1])
                if word_val_caveats[2] != "":
                    if word_val_caveats[2].endswith("1"):
                        word_val_caveats[2] = (0, word_val_caveats[2][:len(word_val_caveats[2]) - 1])
                    else:
                        word_val_caveats[2] = (1, word_val_caveats[2][:len(word_val_caveats[2]) - 1])
                mod_dict[word_val_caveats[0]] = (word_val_caveats[1], word_val_caveats[2])
        mod_file.close()
        mod_file = open("datafile/modifiers_dict.txt", "w")
        mod_file.write(str(mod_dict))
        mod_file.close()

        noun_file = open("datafile/fewer_nouns.txt", "r")
        counter = -1
        noun_dict = {}
        for word in noun_file:
            word = word.strip()
            if word != "":
                counter += 1
                noun_dict[word] = counter
        noun_file.close()
        noun_file = open("datafile/noun_dict.txt", "w")
        noun_file.write(str(noun_dict))
        noun_file.close()

    def tag_dict_and_save_nouns_to_file(self):
        num_words_in_narrowed_dict = 0
        dict_file = open("datafile/narrowed_dict.txt", "r")
        for line in dict_file:
            if line != "":
                num_words_in_narrowed_dict += 1
        dict_file.close()
        if num_words_in_narrowed_dict % 1000 == 0:
            thousands_of_times_to_run = num_words_in_narrowed_dict // 1000
        else:
            thousands_of_times_to_run = (num_words_in_narrowed_dict // 1000) + 1

        dict_file = open("datafile/narrowed_dict.txt", "r")
        noun_file = open("datafile/nouns.txt", "w")

        for i in range(thousands_of_times_to_run):
            list_to_tag = []
            for j in range(1000):
                try:
                    word = dict_file.readline()
                except:
                    break
                word = word.strip()
                if word == "":
                    continue
                list_to_tag.append(word)

            tag_tuples = self.tagger.tag(list_to_tag)

            for word_tuple in tag_tuples:
                if word_tuple[1].startswith('N'):
                    noun_file.write(word_tuple[0] + "\n")
                print(word_tuple)
        noun_file.close()
        dict_file.close()

def main():
    dict_to_narrow = argv[1]
    stop_words = argv[2]
    dn = DictionaryNarrower(dict_to_narrow, stop_words)
    dn.tally_words_from_all_text()
    dn.write_all_words_above_threshold_to_file("datafile/narrowed_dict.txt")
    dn.tag_dict_and_save_nouns_to_file()
    DictionaryNarrower.clean_noun_file()
    DictionaryNarrower.python_readable_files()

if __name__ == "__main__":
    main()