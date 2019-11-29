import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

'''
Part of Speech tagging using CNN and RNN architectures
'''

class POS_Tagging:

	# def __init__(self):
		# data = nltk.corpus.treebank.tagged_sents()
	
	def preprocessing(self, data):
		
		## Access dataset ##
		dataset = data
		print('Size of dataset = ', len(dataset))
		print(dataset[1])
		'''
		[('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), 
		('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), 
		('board', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]
		'''
 		# https://nlpforhackers.io/lstm-pos-tagger-keras/
 		## Seperaste the sentences and pos tags so there are corresponding lists of lisrts of each ##
		sentences = []
		pos_tags = [] 
		for sentence in dataset:
		    word, tag = zip(*sentence)
		    sentences.append(np.array(word))
		    pos_tags.append(np.array(tag))

		print(sentences[1])
		print(pos_tags[1])

		# Find the length of all sentences ands store in list
		sent_len = []
		for i in sentences:
			sent_len.append(len(i))
		max_length = sent_len.index(max(sent_len))
		# print(sentences[1854])
		# print(len(sentences[1854]))

		'''
 		['Pierre' 'Vinken' ',' '61' 'years' 'old' ',' 'will' 'join' 'the' 'board'
 		'Nov.' '29' '.']
		['NNP' 'NNP' ',' 'CD' 'NNS' 'JJ' ',' 'MD' 'VB' 'DT''NN' 'NNP' 'CD' '.']
 		'''

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

		# A dict containing the words and their index
		indexed_words = dict()
		indexed_words = {w: i + 2 for i, w in enumerate(list(words_set))}
		# Add indecies for padding and OOV words
		indexed_words['-PAD-'] = 0
		indexed_words['-OOV-'] = 1

		# A dict for pos tags and their indicies
		indexed_tags = dict()
		indexed_tags = {t: i + 2 for i, t in enumerate(list(tags))}
		indexed_tags['-PAD-'] = 0
		indexed_tags['-OOV-'] = 1

		# For display only
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

		for tags in y_train:
			tag_ints = []
			for tag in tags:
				try:
					tag_ints.append(indexed_tags[tag])
				except KeyError:
					tag_ints.append(indexed_tags['-OOV-'])
			y_train_tags.append(tag_ints)
		print(y_train_tags[0])

		for tags in y_test:
			tag_ints = []
			for tag in tags:
				try:
					tag_ints.append(indexed_tags[tag])
				except KeyError:
					tag_ints.append(indexed_tags['-OOV-'])
			y_test_tags.append(tag_ints)
		print(y_test_tags[0])

		## Pad sequences ##
		X_train_sent = pad_sequences(X_train_sent, maxlen=max_length, padding='post')
		X_test_sent = pad_sequences(X_test_sent, maxlen=max_length, padding='post')
		y_train_tags = pad_sequences(y_train_tags, maxlen=max_length, padding='post')
		y_test_tags = pad_sequences(y_test_tags, maxlen=max_length, padding='post')
		 
		print(X_train_sent[0])
		print(X_test_sent[0])
		print(y_train_tags[0])
		print(y_test_tags[0])

		return X_train_sent, y_train_tags, X_test_sent, y_train_tags, max_length
				
	def CNN():
		



if __name__ == "__main__":

	dataset = nltk.corpus.treebank.tagged_sents()
	extractor = POS_Tagging()
	extractor.preprocessing(dataset)
