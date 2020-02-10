import nltk
import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt

from tensorflow.python.keras.layers import Dense, LSTM, GRU, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.python.keras.optimizers import Adam
from keras import backend as K

'''
Part of Speech tagging using CNN and RNN architectures
'''

class POS_Tagging:

	def __init__(self):
		data = nltk.corpus.treebank.tagged_sents()
	
	def preprocessing(self, data):
		
		## Access dataset ##
		dataset = data#[0:100]
		print('Size of dataset = ', len(dataset))
		print(dataset[1])
	
 		## Seperaste the sentences and pos tags so there are corresponding lists of lisrts of each ##
		sentences = []
		pos_tags = [] 
		for sentence in dataset:
		    word, tag = zip(*sentence)
		    sentences.append(np.array(word))
		    pos_tags.append(np.array(tag))

		print(sentences[5])
		print(pos_tags[5])

		# Find the length of all sentences ands store in list
		sent_len = []
		for i in sentences:
			sent_len.append(len(i))
		
		# MAX_LENGTH = max(sent_len)
		# print(MAX_LENGTH)
		
		# print(sentences[1854])
		# print(len(sentences[1854]))

		## Split into test/train ##
		X_train, X_test, y_train, y_test = train_test_split(sentences, pos_tags, test_size=0.2)

		## Convert word list and pos tag lists to numerical ##
		# Give each unique word an integer
		# COnvert all sentences to lower case
		
		# Create sets of words and tags - to then use to create dictionaries
		words_set = set([])
		tags_set = set([])
		
		for sent in sentences:
			for word in sent:
				words_set.add(word.lower())
		
		for tags in pos_tags:
			for tag in tags:
				tags_set.add(tag)

		# for i in pos_tags: print(i)

		# A dict containing the words and their index
		# indexed_words = dict()
		indexed_words = {w: i + 2 for i, w in enumerate(list(words_set))}
		# Add indecies for padding and OOV words
		indexed_words['-PAD-'] = 0
		indexed_words['-OOV-'] = 1

		length_word_index = len(indexed_words)
		print(length_word_index)
		
		# A dict for pos tags and their indicies
		# indexed_tags = dict()
		indexed_tags = {t: i + 1 for i, t in enumerate(list(tags_set))}
		indexed_tags['-PAD-'] = 0
		# indexed_tags['-OOV-'] = 1

		length_tag_index = len(indexed_tags)
		print(length_tag_index)

		# Display
		N = 10
		output = dict(list(indexed_words.items())[0: N]) 
		print(output)
		output = dict(list(indexed_tags.items())[0: N]) 
		print(output)

		X_train_sent, X_test_sent, y_train_tags, y_test_tags = [], [], [], []
 		
		for sent in X_train:
		    sent_ints = []
		    for word in sent:
		        try:
		            sent_ints.append(indexed_words[word.lower()])
		        except KeyError:
		            sent_ints.append(indexed_words['-OOV-'])
		 
		    X_train_sent.append(sent_ints)
		print(X_train_sent[0])
		 
		for sent in X_test:
		    sent_ints = []
		    for word in sent:
		        try:
		            sent_ints.append(indexed_words[word.lower()])
		        except KeyError:
		            sent_ints.append(indexed_words['-OOV-'])
		 
		    X_test_sent.append(sent_ints)
		print(X_test_sent[0])

		for s in y_train:
			y_train_tags.append([indexed_tags[t] for t in s])
 
		for s in y_test:
			y_test_tags.append([indexed_tags[t] for t in s])


		## Pad sequences ##
		# Find max length of sent
		MAX_LENGTH = len(max(X_train_sent, key=len))
		print(MAX_LENGTH) 

		X_train_sent = pad_sequences(X_train_sent, maxlen=MAX_LENGTH, padding='post')
		X_test_sent = pad_sequences(X_test_sent, maxlen=MAX_LENGTH, padding='post')
		y_train_tags = pad_sequences(y_train_tags, maxlen=MAX_LENGTH, padding='post')
		y_test_tags = pad_sequences(y_test_tags, maxlen=MAX_LENGTH, padding='post')
		 
		# print(X_train_sent[0])
		# print(X_test_sent[0])
		# print(y_train_tags[0])
		# print(y_test_tags[0])

		return (X_train_sent, y_train_tags, X_test_sent, y_test_tags, 
				MAX_LENGTH, length_word_index, length_tag_index)
	
	def CNN(self, X_train, y_train, X_test, y_test, MAX_LENGTH, length_word_index, length_tag_index):

		print('\n=== Convolutional Neural Network ===\n')
		
		print(y_train.shape) #(3131, 272)
		variable = y_train.shape[0]
		# Build CNN model
		model = Sequential()
		model.add(Embedding(input_dim=length_word_index, output_dim=length_tag_index, input_length=MAX_LENGTH))
		# model.add(Conv1D(filters=MAX_LENGTH, kernel_size=4, activation='relu'))
		# model.add(GlobalMaxPooling1D())
		# model.add(InputLayer(input_shape=(MAX_LENGTH,)))
		# model.add(Embedding(input_dim=length_word_index, output_dim=128, input_length=MAX_LENGTH))
		model.add(Conv1D(filters=MAX_LENGTH, kernel_size=4, padding='same', activation='relu'))
		model.add(Dense(length_tag_index))
		model.add(Activation('softmax'))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		print(model.summary())	

		# One hot encode the tags
		def onehot_encode_tags(sequences, categories):
		    cat_sequences = []
		    for seq in sequences:
		        cats = []
		        for item in seq:
		            cats.append(np.zeros(categories))
		            cats[-1][item] = 1.0
		        cat_sequences.append(cats)
		    return np.array(cat_sequences)

		cat_train_tags_y = onehot_encode_tags(y_train, length_tag_index)
		print(cat_train_tags_y[0])

		print('y_train shape: ', y_train.shape)
		print('MAX_LENGTH: ', MAX_LENGTH)
		print('length_word_index: ', length_word_index)
		print('length_tag_index: ', length_tag_index)

		# Train the model
		history = model.fit(X_train, onehot_encode_tags(y_train, length_tag_index), batch_size=128, epochs=1, validation_split=0.2)
		
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, onehot_encode_tags(y_test, length_tag_index))
		print('Accuracy: %f' % (accuracy * 100))

	def LSTM(self, X_train, y_train, X_test, y_test, MAX_LENGTH, length_word_index, length_tag_index):

		model = Sequential()
		model.add(InputLayer(input_shape=(MAX_LENGTH, )))
		model.add(Embedding(length_word_index, 128))
		# model.add(Conv1D(filters=MAX_LENGTH, kernel_size=4, padding='same', activation='relu'))
		model.add(LSTM(256, return_sequences=True))
		model.add(TimeDistributed(Dense(length_tag_index)))
		model.add(Activation('softmax'))
		 
		model.compile(loss='categorical_crossentropy',
		              optimizer=Adam(0.001),
		              metrics=['accuracy'])
		              # metrics=['accuracy', ignore_class_accuracy(0)])

		model.summary()

		# One hot encode the tags
		def onehot_encode_tags(sequences, categories):
		    cat_sequences = []
		    for seq in sequences:
		        cats = []
		        for item in seq:
		            cats.append(np.zeros(categories))
		            cats[-1][item] = 1.0
		        cat_sequences.append(cats)
		    return np.array(cat_sequences)

		cat_train_tags_y = onehot_encode_tags(y_train, length_tag_index)
		print(cat_train_tags_y[0])

		# Train the model
		history = model.fit(X_train, onehot_encode_tags(y_train, length_tag_index), batch_size=228, epochs=1, validation_split=0.2)
		
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, onehot_encode_tags(y_test, length_tag_index))
		print('Accuracy: %f' % (accuracy * 100))

	def bi_LSTM(self, X_train, y_train, X_test, y_test, MAX_LENGTH, length_word_index, length_tag_index):

		model = Sequential()
		model.add(InputLayer(input_shape=(MAX_LENGTH, )))
		model.add(Embedding(length_word_index, 128))
		# model.add(Conv1D(filters=MAX_LENGTH, kernel_size=4, padding='same', activation='relu'))
		model.add(Bidirectional(LSTM(256, return_sequences=True)))
		model.add(TimeDistributed(Dense(length_tag_index)))
		model.add(Activation('softmax'))
		 
		model.compile(loss='categorical_crossentropy',
		              optimizer=Adam(0.001),
		              metrics=['accuracy'])
		              # metrics=['accuracy', ignore_class_accuracy(0)])

		model.summary()

		# One hot encode the tags
		def onehot_encode_tags(sequences, categories):
		    cat_sequences = []
		    for seq in sequences:
		        cats = []
		        for item in seq:
		            cats.append(np.zeros(categories))
		            cats[-1][item] = 1.0
		        cat_sequences.append(cats)
		    return np.array(cat_sequences)

		cat_train_tags_y = onehot_encode_tags(y_train, length_tag_index)
		print(cat_train_tags_y[0])

		# Train the model
		history = model.fit(X_train, onehot_encode_tags(y_train, length_tag_index), batch_size=228, epochs=1, validation_split=0.2)
		
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, onehot_encode_tags(y_test, length_tag_index))
		print('Accuracy: %f' % (accuracy * 100))
		
		# ## Manual test ##
		# test_samples = ["running is very important for me .".split(),
 		#   					"I was running every day for a month .".split()]
		
		# print(test_samples)
		# test_samples_X = []
		# for s in test_samples:
  		#   		s_int = []
  		#   		for w in s:
		#         try:
		#             s_int.append(word2index[w.lower()])
		#         except KeyError:
		#             s_int.append(word2index['-OOV-'])
		#     test_samples_X.append(s_int)
		 
		# test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
		# print(test_samples_X)

	def GRU(self, X_train, y_train, X_test, y_test, MAX_LENGTH, length_word_index, length_tag_index):

		model = Sequential()
		model.add(InputLayer(input_shape=(MAX_LENGTH, )))
		model.add(Embedding(length_word_index, 128))
		model.add(GRU(50, return_sequences=True))
		model.add(GRU(length_tag_index, return_sequences=False))
		model.add(Dense(1, activation='sigmoid'))
		# model.add(Activation('softmax'))
		 
		model.compile(loss='categorical_crossentropy',
		              optimizer=Adam(0.001),
		              metrics=['accuracy'])
		              # metrics=['accuracy', ignore_class_accuracy(0)])

		model.summary()

		# One hot encode the tags
		def onehot_encode_tags(sequences, categories):
		    cat_sequences = []
		    for seq in sequences:
		        cats = []
		        for item in seq:
		            cats.append(np.zeros(categories))
		            cats[-1][item] = 1.0
		        cat_sequences.append(cats)
		    return np.array(cat_sequences)

		cat_train_tags_y = onehot_encode_tags(y_train, length_tag_index)
		print(cat_train_tags_y[0])

		# Train the model
		history = model.fit(X_train, onehot_encode_tags(y_train, length_tag_index), batch_size=228, epochs=1, validation_split=0.2)
		
		# Evaluate the model
		loss, accuracy = model.evaluate(X_test, onehot_encode_tags(y_test, length_tag_index))
		print('Accuracy: %f' % (accuracy * 100))
		
		 

if __name__ == "__main__":

	dataset = nltk.corpus.treebank.tagged_sents()
	extractor = POS_Tagging()
	(X_train, y_train, X_test, y_test, 
	MAX_LENGTH,length_word_index, length_tag_index) = extractor.preprocessing(dataset)
	extractor.CNN(X_train, y_train, X_test, y_test, MAX_LENGTH, length_word_index, length_tag_index)
	extractor.LSTM(X_train, y_train, X_test, y_test, MAX_LENGTH, length_word_index, length_tag_index)
	extractor.bi_LSTM(X_train, y_train, X_test, y_test, MAX_LENGTH, length_word_index, length_tag_index)
	extractor.GRU(X_train, y_train, X_test, y_test, MAX_LENGTH, length_word_index, length_tag_index)




