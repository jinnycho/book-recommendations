# BookRecComps2016
*Book Recommendation Senior Comps for Carleton College 2016.*

There are three major approaches for recommendation systems: content-based, collaborative, and hybrid. Broadly, recommendation systems that implement a content-based (CB) approach recommend items to a user that are similar to the ones the user preferred in the past. On the other hand, recommendation systems that implement collaborative filtering (CF) predict users’ preferences by analyzing relationships between users and interdependencies among items; from these, they extrapolate new associations.  Finally, hybrid approaches meld content-based and collaborative approaches, which have complementary strengths and weaknesses, thus producing stronger results. For our project we've implemented algorithms from each approach. These programs should be ran in python3 and use numpy and scikit/ scipy.

NOTE: datafile/GoogleBooksDescriptions.txt and datafile/GoodReadsDescriptions.txt are to be used purely for educational or personal use.

## Preprocessing Data
For this project, we used two datasets: BookCrossing (BX) and Amazon Book Reviews (AB).
Users don’t tend to rate or read many books. So one can imagine how inherently sparse our data is. In order to alleviate this sparsity, we used the intersection (by books) of these two datasets for our project. Intention being that it will increase the number of ratings each book has. Then the data is partitioned into training, development, and evaluation for testing.
##### 1) Analyze the original datasets
- user_datacollector.py
    - Requires:
       - datafile/AmazonSorted.csv
       - datafile/BXSorted.csv
    - Run with the following command
       `python3 user_datacollector.py`
    - Results:
    Maps a given user, u, to a list of tuples [(ISBN, rating)...]
        - datafile/user_ratings.txt
        - datafile/book_counts_for_users.txt
    Filters out users who've rated less than three books
        - datafile/user_ratings_g3.txt
    Produces partitioned data
        - datafile/training_ratings.txt
        - datafile/development_ratings.txt
        - datafile/evaluation_ratings.txt
    Collects data for analyzing variation in user ratings
        - datafile/user_rating_variation.csv

*NOTE that many of the results are dependent on the prior*

##### 2) Find the intersection of the data
- intersection_explorer.py
    - Requires:
        - datafile/ISBNNumsInCommon.txt
        - reviews_Books_5.json
        - BX-Book-Ratings.csv
        - datafile/training_ratings.txt
    - Run with the following command
        `python3 intersection_explorer.py`
    - Results:
    Creates csv and txt files that stores information about the data we'd like to use
        - BXsubset.csv  and  AmazonSubset.csv
        - datafile/ISBNIntersection.txt


## Collaborative Filtering
##### 1) Creation of Utility Matrix
Creates a utility matrix class that contains useful information about books, users and ratings. Users' ratings are well used in both Collaborative Filtering and Content-baed Filtering.
- dev_data_to_dict.py
    - Requires:
        - datafile/ISBNIntersection.txt
        - datafile/training_ratings.txt
    - Run with the following command
        - `python3 dev_data_to_dict.py`
    - Results:
        - 4 dictionaries that map isbn index to real isbn, isbn to isbn index, user index to real user id, user id to user index.
        - utility matrix of type scipy.csr_matrix that contains users, books and ratings information.
        - the size of the utility matrix is (37563, 98268)
    - The image of the utilty matrix:
        ![util_mat](https://github.com/eunjincho503/Computer_Org_Arch/blob/master/matrix.png "utility matrix")

##### 2) Generating Predictions for Development Data
*NOTE: to generate predictions for evaluation data instead, users need to change input and output filenames/paths in the code accordingly*
- kNN.py
  - Requires:
    - datafile/UtilMat_obj.p
    - datafile/dev_dict.p (when predict for development data)
    - datafile/eval_dict.p (when predict for evaluation data)

  - Run with the following command
     - `python3 kNN.py`  
   - Results:
     - 174 picked files that encode kNN predicted ratings for the development data will be stored in two folders
         - 87 pickled dictionaries will be stored in the folder dev_norm_pred. They correspond to the results obtained after normalizing the training data.
         - 87 pickled dictionaries will be stored in the folder dev_unnorm_pred. They correspond to the results obtained without normalizing the training data.
     - Each file corresponds to a pickled nested python dictionary that encodes rating predictions made using a different combination of parameters.
          - Each key of the dictionary is a user index and its value is an inner dictionary that maps from book indices to predicted ratings
	       - e.g. [u1: [b1:r1, b2:r2, …]]
	  - To access the predicted rating of user u to book b in the dictionary, use `dictionary[u][b]`
		The file name of the pickled dictionary contains information about the similarity measure and the k used.

		
##### 3) Evaluation
- rmse.py
    - Requires:
        - datafile/UtilMat_obj.p
        - datafile/nonzero_training_indices.p
        - datafile/dev_dict.p or eval_dict.p
	- Load dev_dict:
	    dev_dict.p or eval_dict.p are nested dictionaries that contain the actual ratings
	    the actual ratings, comment in whichever dataset you’d prefer
	    to calculate the rmse value for
	- Loading prediction dictionaries:

	    **Development**
	    by the nature of testing there are 174 different combinations
	    that need to be tested in development; switch the comments in
	    function calculate_rmse for normalized or unnormalized

	    **Evaluation**
	    in main comment in “Run rmse for eval” and comment out “Run rmse for dev”
        in calculate_rmse ensure that the filename reflects the 'best point'
	- Saving cumulative_rmse, dev dictionary:
        the cumulative rmse dicts for dev are saved in main, ensure that norm v. unnorm are correctly pickled
        NOTE that the cumulative rmse dicts evalution are saved in calculate_rmse
    - Run with the following command
        `python3 rmse.py`  
    - Results:
        each rmse dict is stored in a directory that denotes whether or not the data was normalized
        for each similiarity metric and k neighbors pair their exists a rmse_single_bin and single_bin_rmse; the rmse_single_bin stores the necessary information to compute the RMSE value, whereas the single_bin_rmse maps the number of ratings given by the training data to the RMSE value, which allows us observe how our predictions do with more or less information

*NOTE: The versions of uvd.py and pca.py are still in development. For further development please checkout the branch 'uvd+pca’*

## Content Based Filtering

##### 1) Initial Data Collection from APIs:
(Can be skipped if GoogleBookDescriptions.txt and GoodReadsDescriptions.txt
are both already in the datafile directory)
- get_description_google.py
    - Collects book descriptions from the Google Books API.
    - Requires:
        - datafile/ISBNNumsInCommonRewritable.txt:
            a text file with one ISBN number per line where it is okay for
            its contents to be gradually deleted
        - an API key for Google Books saved to api_key
          at the top of the file (API key can be obtained from
          https://developers.google.com/books/docs/v1/using#APIKey)
    - Run with the following command
        `python3 get_description_google.py`
    - Results:
        - datafile/GoogleBooksDescriptions[insert date run here].txt:
            a text file with one ISBN number and description per line
            containing up to 1000 descriptions from the Google Books API,
            corresponding to the first 1000 ISBNs from
            datafile/ISBNNumsInCommonRewritable.txt.
            Eventually, these can all be manually combined into one file.
            (The GoogleBooks API allows 1000 calls per day on a given
            developer key.)
        - Removes the first 1000 ISBNs stored in
          datafile/ISBNNumsInCommonRewritable.txt
- get_description_gr.py
    - Collects book descriptions from the Goodreads API.
    - Requires:
        - datafile/ISBNNumsInCommon.txt:
            a text file with one ISBN number per line
        - an API key for Goodreads saved to api_key at the top of the
          file (API key can be obtained from https://www.goodreads.com/api)
    - Run with the following command:
        `python3 get_description_gr.py`
    - Results:
        - datafile/GoodReadsDescriptions.txt:
            a text file with one ISBN number and description per line

##### 2) Individual Book Text Aggregation:
Groups all text about the books into their own text files by book,
and also cleans and lemmatizes the words and removes punctuation.
This takes a long time (up to a few days for around 9,000 books)
to run, so it's recommended to run this several times in parallel
with different ISBN files as arguments.
- book_text_files_maker.py
    - Aggregates all text about a given book into a single text file
    (does this for all books)
    - New requirements this step:
        - datafile/Books_5.json:
          a file of JSON dictionaries (sorted by ISBNs) where each
          dictionary (which is its own line) contains the key "reviewText"
          and "asin" (where the "asin" field is the ISBN).
          We used Amazon review data obtained from the following URL:
          http://jmcauley.ucsd.edu/data/amazon/links.html
        - a directory called isbn_text_files (or something else,
          as long as the variable text_file_directory_name is changed
          to reflect that)
        - NLTK. Download using following terminal command:
          `sudo pip3 install -U nltk`
        - Stanford POS Tagger. Download from
            http://nlp.stanford.edu/software/tagger.shtml
          and replace POS Tagger directory names at top of file if needed
    - Requirements from previous steps:
        - datafile/GoodReadsDescriptions.txt, generated above
        - datafile/GoogleBooksDescriptions.txt, generated above
    - Run with the following command:
        `python3 book_text_files_maker.py filename_of_file_containing_ISBNs`
    - Results:
        - isbn_text_files (or replaced text_file_directory_name) will be
        filled with text files, one per book, where each text file contains
        a Python list of lists of strings, which are each a single lowercase
        lemmatized word (one inner list per sentence)

##### 3) Dictionary Generation:
Makes a few versions of a narrowed dictionary of words, based on which
words appear at least a certain number of times in our lemmatized text
about books from step 2.
- dictionary_narrower.py
    - New requirements this step:
        - datafile/SentDict.csv
            a CSV file with words in the first field, and numbers
            representing their (non-neutral) sentiment values in the second
            field
        - datafile/stop_words.txt
            a text file with one stop word per line
        - datafile/modifiers_hinges_plain.txt (included in repo)
            a text file with one modifier or hinge word per line
        - datafile/modifiers_vals.txt (included in repo)
            includes modifiers' values, as well as any words coming before
            or after them that would make them not a modifier (and a 1 if
            that extra word would be coming before, or a 2 if after)
    - Requirements from previous steps:
        - Stanford POS Tagger. Download from
            http://nlp.stanford.edu/software/tagger.shtml
          and replace POS Tagger directory names at top of file if needed
        - NLTK
        - datafile/ISBNNumsInCommon.txt
            or a different text file of ISBNs, with the isbn_file variable
            at the top of the file changed to reflect that
        - isbn_text_files (or text_file_directory_name from
          book_text_files_maker.py) to be a directory full of the lemmatized,
          cleaned text files created in step 2
    - Run with the following command:
        `python3 dictionary_narrower.py original_dictionary stopwords_filename`
    - Results:
        - datafile/narrowed_dict.txt
            one word per line. Only includes words from original_dictionary
            that occurred more than threshold (the variable) times across
            all lemmatized book text.
        - datafile/nouns.txt
            all nouns from narrowed_dict.txt
        - datafile/fewer_nouns.txt
            just the nouns that aren't opinion words, stop words, modifiers,
            or hinges
        - datafile/noun_dict.txt
            text with python-readable code representing a dictionary
            mapping nouns to their indices
        - datafile/modifiers_dict.txt
            text with python-readable code mapping modifiers to values
            in a dictionary

##### 4) Book Vector Generation:
Makes the modeled-book matrices using the generated dictionaries and
cleaned, lemmatized text about the books.
- keyword_vector_maker.py
    - New requirements this step:
        - scipy.sparse
        - numpy
    - Requirements from previous steps:
        - datafile/narrowed_dict.txt, from step 3
        - datafile/ISBNNumsInCommon.txt, from step 1
        - isbn_text_files (or whatever text_file_directory_name in
          book_text_files_maker.py is), which should be full of python-
          readable lemmatized text as lists
    - Run with the following command:
        `python3 keyword_vector_maker.py`
    - Results:
        - datafile/TFIDF.npz
            a sparse representation of the resulting matrix
- sentiment_vector_maker.py
    - New requirements this step:
        - scipy.sparse
        - numpy
    - Requirements from previous steps:
        - datafile/noun_dict.txt, from step 3
        - datafile/modifiers_dict.txt, from step 3
        - datafile/ISBNNumsInCommon.txt, from step 1
        - isbn_text_files (or whatever text_file_directory_name in
          book_text_files_maker.py is), which should be full of python-
          readable lemmatized text as lists
    - Run with the following command:
        `python3 sentiment_vector_maker.py`
    - Results:
        - datafile/Sentiment.npz
            a sparse representation of the resulting matrix

##### 5) Running Tests and Generating Predictions:
Makes predictions for users using their training data (and runs the
high-level tests involving batches of many users)
- UserProfileLearner/user_profile_learner.py
    - New requirements this step:
        - sklearn.linear model
            (download from http://scikit-learn.org/stable/install.html)
        - training ratings by users, provided in variable at top of file
          called training_data_filename
        - ratings by users to train classifier to match, provided
          in evaluation_data_filename
        - datafile/isbns_to_indices_dict.txt
        - datafile/indices_to_isbns_dict.txt
    - Requirements from previous steps:
        - scipy.sparse
        - numpy
        - datafile/TFIDF.npz
        - datafile/Sentiment.npz
        - datafile/ISBNNumsInCommon.txt
        - datafile/narrowed_dict.txt
        - datafile/fewer_nouns.txt
    - Run with the following command:
        `python3 user_profile_learner.py`
                {tfidf, sentiment, both_indep, both_by_tfidf}
                {naive_bayes, maxent}
                num_features_to_use
    - Results:
        - datafile/results_by_ratings_[classifier_type]_[data_type]_
          [num_feats]_[feat_selection_method]_[date/time].csv
            a csv file of results produced for each predicted rating
        - datafile/results_by_users_[classifier_type]_[data_type]_
          [num_feats]_[feat_selection_method]_[date/time].csv
            a csv file of results produced, with information summarizing
            results of a single user per line

## Hybrid
##### 1) Convert CF results to binary
Because the CF side predicts ratings for books on a 1 to 5 scale, in order to create a
hybrid system we must first convert the predictions to a binary scale 1 (good book) or
0 (bad book). This is determined by a threshold rating of 2.5, which is to say a book with a rating of 2.5 and below is 'bad' and above 2.5 is 'good'. Once this information is
converted, we calculate the confidence and store it in the same format (nested dict).
- data_for_hybrid.py
    - Requires:
        - datafile/UtilMat_obj.p
        - best point from the eval file (the script as it is computes more information than it needs)
    - Run with the following command
        `python3 data_for_hybrid.py`  
    - Results:
        - datafile/cf_confidence_dict_eval.p

##### 2) Hybrid
For this particular dataset the weights are 50:50 from CB:CF. This may change depending on one's results.
- hybrid.py
    - Requires:
        - datafile/UtilMat_obj.p
        * Data from best combinations on the evaluation dataset for both CF & CB *
        - datafile/cf_confidence_dict_eval.p
        - maxent_bothbytfidf_100feats_unpairedfeats_2-23_12-52.csv
    - Run with the following command:
    `python3 hybrid.py`  
    - Results:
        - datafile/hybrid_results_(day)_(time)
