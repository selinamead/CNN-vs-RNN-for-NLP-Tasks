
''' Machine Translation of German to English, English to German using CNN and RNN architectures '''
# import csv
import string
import re
import numpy as np
import pandas as pd 
from numpy import array, argmax, random, take

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers

from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Import CNN Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Reshape, TimeDistributed
from tensorflow.python.keras.layers import Input, InputLayer, Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, Embedding, Reshape, TimeDistributed, Activation, RepeatVector
import matplotlib.pyplot as plt
# IMport RNN Libraries
from tensorflow.python.keras.layers import LSTM, Bidirectional, GRU
from tensorflow.python.keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

class Machine_Translation:

	def __init__(self, file_path):
		self.file_path = file_path
		self.size = 5000 # How many samples to use

	def get_dataset(self):

		print('Retrieving Eng-Ger Sentences...')
		file = open(self.file_path, mode='rt', encoding='utf-8')
		data = file.read()
		file.close()
		
		return data
		

	def pre_process(self, data):

		dataset = data
		# print(dataset[:10])
		# Smile.	Lächeln!	CC-BY 2.0 (France) Attribution: tatoeba.org #2764108 (CK) & #4659632 (AC)
		# Need to split on \t and \n
		sentences = dataset.split('\n')
		sentences = [i.split('\t') for i in sentences]
		sentences = array(sentences)
		# for i in range(50): print(sentences[i])
		# ['Fire!' 'Feuer!''CC-BY 2.0 (France) Attribution: tatoeba.org #1829639 (Spamster) & #1958697 (Tamy)']
		eng_sent = []
		ger_sent = []
		for sent in range(self.size):
			# print(sentences[sent][0])
			# print(sentences[sent][1])
			eng = sentences[sent][0]
			ger = sentences[sent][1]
			
			eng_sent.append(eng)
			ger_sent.append(ger)

		print(eng_sent[100]) # Can I help?
		print(ger_sent[100]) # Kann ich mich nützlich machen?

		eng_sent = array(eng_sent) 
		# for i in range(10): print(eng_sent[i])

		# print(eng_sent) # ['Hi.' 'Hi.' 'Run!' ... 'I miss my cat.' 'I miss my mom.' 'I must fix it.']
		
		# Remove puction
		# Source: https://www.analyticsvidhya.com/blog/2019/01/neural-machine-translation-keras/
		eng_sent = [s.translate(str.maketrans('', '', string.punctuation)) for s in eng_sent]
		# print(eng_sent) # 'Hi', 'Hi', 'Run', 'Wow', 'Wow', 'Fire', 'Help', 'Help', 'Stop'
		ger_sent = [s.translate(str.maketrans('', '', string.punctuation)) for s in ger_sent]

		length = len(eng_sent)
		# Convert to lower case
		for i in range(length):
			eng_sent[i] = eng_sent[i].lower()
			ger_sent[i] = ger_sent[i].lower()

		# print(eng_sent)

		# Find max length of sentences
		
		# length_eng, length_ger = [], []
		# for i in eng_sent:
		# 	length_eng.append(len(i.split()))
		# for i in ger_sent:
		# 	length_ger.append(len(i.split()))
		
		max_eng_length = len(max(eng_sent, key=len))
		max_ger_length = len(max(ger_sent, key=len))
		print('Max length of English Sentence: ', max_eng_length)
		print('Max length of German Sentence: ', max_ger_length)

		print(eng_sent.index(max(eng_sent)))
		print(ger_sent.index(max(ger_sent)))
		# print('Longest English Sentence:', eng_sent[1732])
		# print('Longest German Sentence:', ger_sent[1006])
			
		# function to build a tokenizer
		def tokenization(lines):
			tokenizer = Tokenizer()
			tokenizer.fit_on_texts(lines)
			return tokenizer
      	
      	# prepare english tokenizer
		eng_tokenizer = tokenization(eng_sent)
		eng_vocab_size = len(eng_tokenizer.word_index) + 1 # add 1 for 0 indexing
		print('English Vocabulary Size: %d' % eng_vocab_size)
		eng_length = 8

		# prepare Deutch tokenizer
		ger_tokenizer = tokenization(ger_sent)
		ger_vocab_size = len(ger_tokenizer.word_index) + 1 # add 1 for 0 indexing
		print('German Vocabulary Size: %d' % ger_vocab_size)
		ger_length = 11
		max_len = 11

		# encode and pad sequences
		def encode_sequences(tokenizer, length, lines):
	         # integer encode sequences
	         seq = tokenizer.texts_to_sequences(lines)
	         # pad sequences with 0 values
	         seq = pad_sequences(seq, maxlen=length, padding='post')
	         return seq

		# Split into train and test
		# German will be X, English will be y
		X_train, X_test, y_train, y_test = train_test_split(ger_sent, eng_sent, test_size=0.2, random_state=42)

		print(X_train[0],':', y_train[0])
		print(X_test[0], ':', y_test[0])
		
		# Convert text to numerical data. Each unique word has a corresponding integer assigned
		# tokenizer = Tokenizer(oov_token = True)
		# tokenizer.fit_on_texts(X_train)
		# tokenizer.fit_on_texts(y_train)
		
		# X_train = tokenizer.texts_to_sequences(X_train)
		# y_train = tokenizer.texts_to_sequences(y_train)
		
		# X_test = tokenizer.texts_to_sequences(X_test)
		# y_test = tokenizer.texts_to_sequences(y_test)

		# print(X_train[0],':', y_train[0])
		# print(X_test[0], ':', y_test[0])

		# # Padding
		# max_len = 11
		# X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
		# y_train = pad_sequences(y_train, padding='post', maxlen=max_len)

		# X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
		# y_test = pad_sequences(y_test, padding='post', maxlen=max_len)
		
		# print(X_train[0])
		# print(y_train[0])
		# print(X_test[0])
		# print(y_test[0])

		# Bidirection needs same length sequeneces (change padding length of eng to ger)
		X_train = encode_sequences(ger_tokenizer, ger_length, X_train)
		y_train = encode_sequences(eng_tokenizer, ger_length, y_train)
		X_test = encode_sequences(ger_tokenizer, ger_length, X_test)
		y_test = encode_sequences(eng_tokenizer, ger_length, y_test)

		print(X_train[0])
		print(y_train[0])
		print(X_test[0])
		print(y_test[0])

		return X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len

	def CNN(self, X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len):

		units = 128
		model = Sequential()
		model.add(Embedding(input_dim=ger_vocab_size, output_dim=units, input_length=max_len))
		model.add(Conv1D(filters=max_len, kernel_size=4, activation='softmax'))
		# model.add(GlobalMaxPooling1D())
		model.add(Dense(eng_vocab_size, activation='softmax'))
		model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

		model.summary()

		# Train model
		# history = model.fit(X_train, y_train, batch_size=100, epochs=3, verbose=1, validation_split=0.2)
		history = model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1), epochs=3, batch_size=units, verbose=1, validation_split=0.2)
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), verbose=1)
		print('Accuracy: %f' % (accuracy * 100))


	def LSTM(self, X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len):

		print(X_train.shape)
		print(y_train.shape)
		print(X_test.shape)
		print(y_test.shape)

		units = 128
		model = Sequential()
		model.add(Embedding(input_dim=ger_vocab_size, output_dim=units, input_length= 11))
		model.add(LSTM(units))
		model.add(RepeatVector(8))
		model.add(LSTM(units, return_sequences=True))
		model.add(Dense(eng_vocab_size, activation='softmax'))
		model.compile(optimizer=RMSprop(lr=0.01), 
					  loss='sparse_categorical_crossentropy', 
					  metrics=['acc'])

		model.summary()

		# Train model
		history = model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1), epochs=3, batch_size=units, verbose=1, validation_split=0.2)
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), verbose=1)
		print('Accuracy: %f' % (accuracy * 100))

	def bi_LSTM(self, X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len):

		print(X_train.shape)
		print(y_train.shape)
		print(X_test.shape)
		print(y_test.shape)

		units = 128
		model = Sequential()
		model.add(Embedding(input_dim=ger_vocab_size, output_dim=units, input_length=max_len))
		model.add(Bidirectional(LSTM(units, return_sequences=True)))
		# model.add(TimeDistributed(Dense(eng_vocab_size)))
		# model.add(Activation('softmax'))
		model.add(Dense(eng_vocab_size, activation='softmax'))
		model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

		model.summary()

		# Train model
		history = model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1), epochs=3, batch_size=units, verbose=1, validation_split=0.2)
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), verbose=1)
		print('Accuracy: %f' % (accuracy * 100))

		# Bidirection needs same length sequeneces (change padding length of eng to ger)
		
	def GRU(self, X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len):

		units = 128
		model = Sequential()
		model.add(Embedding(input_dim= ger_vocab_size, output_dim=units, input_length=max_len))
		model.add(GRU(units, return_sequences=True))
		# model.add(GRU(eng_vocab_size, return_sequences=False))
		model.add(Dense(eng_vocab_size, activation='softmax'))
		# model.add(Activation('softmax'))
		 
		model.compile(loss='sparse_categorical_crossentropy',
		              optimizer=RMSprop(lr=0.01),
		              metrics=['accuracy'])
		             

		model.summary()

		print(model.summary())	

		# # Train the model
		# history = model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=1, validation_split=0.2)
		# # Evaluate the model
		# loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
		# print('Accuracy: %f' % (accuracy * 100))

		# Train model
		history = model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1), epochs=3, batch_size=units, verbose=1, validation_split=0.2)
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), verbose=1)
		print('Accuracy: %f' % (accuracy * 100))



if __name__ == "__main__":

    file = '/Users/selina/Code/Python/SSL/CNN_vs_RNN/eng_ger_sent.txt'
    extractor = Machine_Translation(file)
    data = extractor.get_dataset()
    # extractor.pre_process(data)
    X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len = extractor.pre_process(data)
    # extractor.CNN(X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len)
    # extractor.LSTM(X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len)
    # extractor.bi_LSTM(X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len)
    extractor.GRU(X_train, y_train, X_test, y_test, eng_vocab_size, ger_vocab_size, max_len)




