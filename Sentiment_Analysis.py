''' Sentiment Analysis of Movie Reviews using CNN, RNN and LSTM architecture '''

import warnings
# Import Pre-Processing libraries
import csv
import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Import CNN Libraries
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D


class Sentiment_Analysis:

	warnings.filterwarnings('ignore')

	def __init__(self, file_path):
		self.file_path = file_path 
		print('Retrieving IMBD Dataset')
		self.dataset = pd.read_csv(file_path)
		self.small_dataset = self.dataset[:10000]


	def pre_process(self, file):
		dataset = self.dataset
		print('Pre Processing Data')
		# Check for null values and drop if any
		if dataset.isnull().values.any():
			print('removing null values')
			dataset = dataset.dropna(how='any', axis=0)
		
		# Get Movie Reviews for cleaning
		small = True # Using small dataset until all code is working
		if small:
			movie_reviews = self.small_dataset
			print('Size of Dataset: ', movie_reviews.shape[0])
		else:
			movie_reviews = self.dataset
			print('Size of Dataset: ', movie_reviews.shape[0])

		# print(movie_reviews.review[7])
	
		# Clean textual data from 'Review'
		stemmer = SnowballStemmer('english') 
		# Function to clean up requirements
		# https://stackoverflow.com/questions/54396405/how-can-i-preprocess-nlp-text-lowercase-remove-special-characters-remove-numb
		def preprocess(sentence):
			sentence = str(sentence)
			sentence = sentence.lower()
			clean_re = re.compile('<.*?>')
			cleantext = re.sub(clean_re, '', sentence)
			rmv_num = re.sub('[0-9]+', '', sentence)
			tokenizer = RegexpTokenizer(r'\w+')
			tokens = tokenizer.tokenize(rmv_num)
			filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('german')]
			stem_words = [stemmer.stem(w) for w in filtered_words]
			
			return " ".join(filtered_words)

		movie_reviews['Review'] = movie_reviews['review'].map(lambda s:preprocess(s))
		# print(movie_reviews.Review[7])

		# Store cleaned reviews into a list to use for train/test
		X = []
		reviews = list(movie_reviews.Review)
		for review in reviews:
			X.append(review)

		# Convert y(sentiment) column to numerical
		y = []
		sentiments = list(movie_reviews.sentiment)
		for sent in sentiments:
			if sent == 'positive':
				y.append(1)
			else:
				y.append(0)

		# Split into train and test
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# Convert text to numerical data
		# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
		
		'''
		word_counts: A dictionary of words and their counts.
		word_docs: A dictionary of words and how many documents each appeared in.
		word_index: A dictionary of words and their uniquely assigned integers.
		document_count:An integer count of the total number of documents that were used to fit the Tokenizer.
		'''
		# Convert text to numerical data
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(X_train)
		X_train = tokenizer.texts_to_sequences(X_train)
		X_test = tokenizer.texts_to_sequences(X_test)
		
		# Number of unique words
		vocab_size = len(tokenizer.word_index) + 1  # need to add 1 because of 0 indexing
		print('Number of Unique Words in Corpus: ', vocab_size)

		# Padding
		maxlen = 100 # Test on 100
		X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
		X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
		
		return X_train, y_train, X_test, y_test, vocab_size

	def CNN(self, X_train, y_train, X_test, y_test, vocab_size):
		print('Convolutional Neural Network')

		X_train = X_train
		X_test = X_test
		y_train = y_train
		y_test = y_test

		# Max length of review for embedding
		max_length = 100 
		embedding_dim = 10
		
		# Define CNN model
		model = Sequential()
		model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
		# model.add(Flatten())
		model.add(Conv1D(128, 10, activation='relu'))
		model.add(GlobalMaxPooling1D())
		model.add(Dense(1, activation='sigmoid'))
		# model.add(Flatten())
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
		print(model.summary())	

		# Train the model
		model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1)
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
		print('Accuracy: %f' % (accuracy*100))
	
	# def RNN():

	# def LSTM():

	# CNN()


if __name__ == "__main__":

    # main() 
    file = '/Users/selina/Code/Python/SSL/CNN_vs_RNN/IMDB_Dataset.csv'
    extractor = Sentiment_Analysis(file)
    X_train, y_train, X_test, y_test, vocab_size = extractor.pre_process(file)
    extractor.CNN(X_train, y_train, X_test, y_test, vocab_size)
    # print(extractor)


