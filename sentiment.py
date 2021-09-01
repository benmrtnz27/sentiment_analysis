"""Applying Machine Learning to Sentiment Analysis"""
"""In this chapter we will delve into a subfield of Natural Language Processing (NLP) called sentiment analysis and learn how to use machine learning algorithms to classify documents based on their polarity: the attitude
of the write. In particular, we are going to work with a dataset of 50,000 movie reviews from the Internet Movie Database (IMDb) and build a predictor that can distinguish between positive and negative reviews."""

"""Sentiment analysis, sometimes called opinion mining, is a popular subdiscipline of the broader field of NLP; it is concerned with analyzing the polarity of documents. A popular task in sentiment analysis is the classification
of documents based on the expressed opinions or emotions of the authors with regard to a particular topic."""

"""preprocessing the movie dataset into more convenient format"""
import os
import sys
import tarfile
import time
import pyprind
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
import pickle


# change the 'basepatch' to the directory of the unzipped movie dataset
basepath = r'C:\Users\benmr\PycharmProjects\MacineLearningBook\sentiment_analysis_main\aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file),
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

"""We initialized a new progress bar object pbar with 50,000 iterations, the same number of documents we were going to read in. Using the nested for loops, we iterated over the train and test subdirectories in the main
aclImdb directory and read the individual text files from pos and neg subdirectories that we eventually appended to the df pandas DataFrame, together with an integer class label (1 = positive, and 0 = negative)."""

"""Since the class labels in the assembled dataset are sorted we will now shuffle DataFrame using the permutation function from the np.random submodule - this will be useful to split the dataset into training and test sets
later sections when we will stream the data from our local drive directly. For our own convenience, we will also store the assembled and shuffled movie review dataset as a CSV file:"""

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))


# Optional: Saving the assembled data as CSV file:



df.to_csv('movie_data.csv', index=False, encoding='utf-8')





df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)


# ### Note
#
# If you have problems with creating the `movie_data.csv` file in the previous chapter, you can find a download a zip archive at
# https://github.com/rasbt/python-machine-learning-book-2nd-edition/tree/master/code/ch08/


# # Introducing the bag-of-words model

# ...

# ## Transforming documents into feature vectors

# By calling the fit_transform method on CountVectorizer, we just constructed the vocabulary of the bag-of-words model and transformed the following three sentences into sparse feature vectors:
# 1. The sun is shining
# 2. The weather is sweet
# 3. The sun is shining, the weather is sweet, and one and one is two
#

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)


# Now let us print the contents of the vocabulary to get a better understanding of the underlying concepts:

print(count.vocabulary_)


# As we can see from executing the preceding command, the vocabulary is stored in a Python dictionary, which maps the unique words that are mapped to integer indices. Next let us print the feature vectors that we just created:

# Each index position in the feature vectors shown here corresponds to the integer values that are stored as dictionary items in the CountVectorizer vocabulary. For example, the first feature at index position 0 resembles the count of the word and, which only occurs in the last document, and the word is at index position 1 (the 2nd feature in the document vectors) occurs in all three sentences. Those values in the feature vectors are also called the raw term frequencies: *tf (t,d)*—the number of times a term t occurs in a document *d*.



print(bag.toarray())



# Assessing word relevancy via term frequency-inverse document frequency



np.set_printoptions(precision=2)


# When we are analyzing text data, we often encounter words that occur across multiple documents from both classes. Those frequently occurring words typically don't contain useful or discriminatory information. In this subsection, we will learn about a useful technique called term frequency-inverse document frequency (tf-idf) that can be used to downweight those frequently occurring words in the feature vectors. The tf-idf can be determined as the product of the term frequency and the inverse document frequency:

# Scikit-learn implements yet another transformer, the `TfidfTransformer`, that takes the raw term frequencies from `CountVectorizer` as input and transforms them into tf-idfs:


tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())


# As we saw in the previous subsection, the word is had the largest term frequency in the 3rd document, being the most frequently occurring word. However, after transforming the same feature vector into tf-idfs, we see that the word is is
# now associated with a relatively small tf-idf (0.45) in document 3 since it is
# also contained in documents 1 and 2 and thus is unlikely to contain any useful, discriminatory information.

# While it is also more typical to normalize the raw term frequencies before calculating the tf-idfs, the `TfidfTransformer` normalizes the tf-idfs directly.

# By default (`norm='l2'`), scikit-learn's TfidfTransformer applies the L2-normalization, which returns a vector of length 1 by dividing an un-normalized feature vector *v* by its L2-norm:

# To make sure that we understand how TfidfTransformer works, let us walk
# through an example and calculate the tf-idf of the word is in the 3rd document.

# The word is has a term frequency of 3 (tf = 3) in document 3, and the document frequency of this term is 3 since the term is occurs in all three documents (df = 3). Thus, we can calculate the idf as follows:

# Now in order to calculate the tf-idf, we simply need to add 1 to the inverse document frequency and multiply it by the term frequency:

tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)

tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]

l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))

# Cleaning text data

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

preprocessor(df.loc[0, 'review'][-50:])

preprocessor("</a>This :) is :( a test :-)!")

df['review'] = df['review'].apply(preprocessor)


# Processing documents into tokens
"""Now we need to split the text into individual elements. We do this by the process of tokenize. Tokenizing documents is to split them into individual words by splitting the cleaned documents at its whitespace characters"""

def tokenizer(text):
    return text.split()
tokenizer('runners like running and thus they run')

"""In the context of tokenization, another useful technique is word stemming, which is the process of transforming a word into its root form. it allows us to map related words to the same stem. The original stemming algorithm was developed by Martin F Porter in 1979, and
is known as the porter stemmer algorithm. We can incorporate this stemming algorithm by importing the PorterStemmer function from the nltk.stem.porter library."""

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer_porter('runners like running and thus they run')

# Stop word-removal
"""Stop words are simply those words that are extremely common in all sorts of texts and bear no useful info that can be used to distinguish different classes of documents. Examples of stop-words are 'is', 'and', 'has', and 'like'. To remove stopwords from the movie reviews,
we will use the set of 127 English stop-words that is available from the NLTK library which can be obtained by calling the nltk.download function."""

nltk.download('stopwords')
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]


# Training a logistic regression model for document classification

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

"""Next we will use a GridSearchCV object to find the optimal set of parameters for our logistic regression model using 5-fold stratified cross-validation"""

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

# gs_lr_tfidf.fit(X_train, y_train)
# print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
# comment out the two above lines because executing they take a long time and only need to be ran once

"""As we can see in the preceding output, we obtained the best grid search results using the regular tokenizer without Porter stemming, no stop-word library, and tf-idfs in combination with a
logistic regression classifier that uses L2-regularization with the regularization strength C of 10.0"""
"""Using the best model from this grid search, let's print the average 5-fold cross-validation accuracy scores on the training set and the classification accuracy on the test dataset."""

# print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
# clf = gs_lr_tfidf.best_estimator_
# print('Test Accuracy: %.3f'
#      % clf.score(X_test, y_test))
# comment these out as well

"""The results reveal that our machine learning model can predict whether a movie review is positive or negative with 90 percent accuracy"""

# Working with bigger data - online algorithms and out-of-core learning
"""If you executed the code examples in the previous section, you may have noticed that it could be computationally quite expensive to construct the feature vectors for the 50,000 movie review dataset
during grid search. In many real-world applications, it is not uncommon to work with even larger datasets that can exceed our computer's memory. Since not everyone has access to supercomputer facilities
we will now apply a technique called out-of-core learning, which allows us to work with such large datasets by fitting the classifier incrementally on smaller batches of the dataset.
Back in Chapter 2, we introduced the concept of stochastic gradient descent, which is an optimization algorithm that updates the model's weights using one sample at a time. In this section, we will
make use of the partial_fit function of the SGDClassifier in scikit-learn to stream the documents directly from our local drive, and train a logistic regression model using small mini-batches of
documents.
First we define a tokenizer function that cleans the unprocessed text data from the movie_data.csv file that we constructed at the beginning of this chapter and separate it into word tokens while
removing stop words."""
import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

"""Next we define a generator function stream_docs that reads in and returns one document at a time"""
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

"""To verify that our stream_docs function works correctly, let's read in the first document from the movie_data.csv file, which should return a tuple consisting of the review text as well as the
corresponding class label"""
print(next(stream_docs(path='movie_data.csv')))

"""We will now define a function, get_minibatch, that will take a document stream from the stream_docs function and return a particular number of documents specified by the size parameter:"""
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None

    return docs, y

"""Unfortunately, we can't use CountVectorizer for out-of-core learning since it requires holding the complete vocabulary in memory. Also, TfidfVectorizer needs to keep all the feature vectors of the
training dataset in memory to calculate the inverse document frequencies. However, another useful vectorizer for text processing implemented in scikit-learn is HashingVectorizer. HashingVectorizer is
data-independent amd makes use of the hashing trick via the 32-but MurmurHash3 function by Austin Appleby"""

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter_no_change=1)
doc_stream = stream_docs(path='movie_data.csv')

"""Using the preceding code, we initialized HashingVectorizer with our tokenizer function and set the number of features to 2**21. Furthermore, we reinitialized a logistic regression classifier by
setting the loss parameter of the SGDClassifier to 'log' - note that by choosing a larger number of features in the HashingVectorizer, we reduced the chance of causing hash collisions, but we also
increase the number of coefficients in our logistic regression model. Now comes the really interesting part. Having set up all the complementary functions, we can now start the out-of-core learning
using the following code:"""

import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

"""Again, we made use of the PyPrind package in order to estimate the progress of our learning algorithm. We initialized the progress bar object with 45 iterations and, in the following for loop, we
iterated over 45 mini-batches of documents where each mini-batch consists of 1,000 documents. Having completed the incremental learning process, we will use the last 5,000 documents to evaluate the
performance of our model:"""
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

"""As we can see, the accuracy of the model is approximately 88 percent, slightly below the accuracy that we achieved in the previous section using the grid search for hyperparameter tuning. However
out-of-core learning is very memory efficient and took less than a minute to complete. Finally, we can use the last 5,000 documents to update our model:"""
clf = clf.partial_fit(X_test, y_test)

# LDA with scikit-learn
"""In this subsection, we will use the LatentDirichletAllocation class implemented in scikit-learn to decompose the movie review dataset and categorize it into different topics. In the following
example, we restrict the analysis to 10 different topics, but readers are encouraged to experiment with the hyperparameters of the algorithm to explore the topics that can be found in this dataset
further."""
"""First, we are going to load the dataset into a pandas DataFrame using the local movie_data.csv file of the movie reviews that we have created at the beginning of this chapter"""

import pandas as pd
df = pd.read_csv('movie_data.csv', encoding='utf-8')

"""Next we are going to use the already familiar CountVectorizer to create the bag-of-words matrix as input to the LDA. For convenience, we will use scikit-learn's built-in English stop word library
via stop_words='english':"""

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)

"""Notice that we set the maximum document frequency of words to be considered to 10 percent (max_df=.1) to exclude words that occur too frequently across documents. The rationale behind the removal
of frequently occurring words is that these might be common words appearing across all documents and are therefore less likely associated with a specific topic category of a given document. Also, we
limited the number of words to be considered to the most frequently occurring 5,000 words (max_features=5000) to limit the dimensionality of this dataset so that it improves the inference performed
by LDA. However, both max_df=.1 and max_features=5000 are hyperparameter values that were chosen arbitrarily."""

"""The following code example demonstrates how to fit a LatentDirichletAllocation estimator to the bag-of-words matrix and infer the 10 different topics from the documents (note that the model fitting
can take up to five minutes or more on a laptop or standard pc)."""
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)

"""By setting learning_method='batch', we let the lda estimator do its estimation based on all available training data (the bag-of-words matrix) in one iteration, which is slower than the alternative
'online' learning method but can lead to more accurate results (setting learning_method='online' is analogous to online or mini-batch learning that we discussed in Chapter 2."""
"""After fitting the LDA, we now have access to the components_attribute of the lda instance, which stores a matrix containing the word importances(here, 5000) for each of the 10 topics in increasing
order:"""
print(lda.components_.shape)

"""To analyze the results, let's print the five most important words for each of the 10 topics. Note that the word importance values are ranked in icreasing order. Thus, to print the top five words,
we need to sort the topic array in reverse order:"""

n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))

"""Based on reading the five most important words for each topic, we may guess that the LDA identified the following topics:
    1. Generally bad movies (not really a topic category)
    2. Movies about families
    3. War movies
    4. Art movies
    5. Crime movies
    6. Horror movies
    7. Comedy movies
    8. Movies somehow related to TV shows
    9. Movies based on books
    10. Action movies
To confirm that the categories make sense based on the reviews, let's plot three movies from the horror movie catergory (horror movies belong to category 6 at index position 5)"""

horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')

"""Using the preceding code example, we printed the first 300 characters from the top 3 horror movies, and we can see that the reviews - even though we don;t know which exact movie they belong to -
sound like reviews of horror movies (however one might argue that Horror movie #2 could also be a good fit for topic category 1L Generally bad movies)."""
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop,
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf,
            open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
