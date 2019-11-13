import string
import tensorflow as tf
import numpy as np
from numpy import array

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Embedding, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from preprocess import Extractor

ext = Extractor()		
ext.read_captions('../data/Flickr8k_text/Flickr8k.token.txt')
ext.read_images('../data/Flicker8k_Dataset/')
ext.get_stats()

captions, images = array(ext.find_pairs())
  
tokenizer = Tokenizer(filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(captions)

vocab_size = len(tokenizer.word_index) + 1
seq_len = 12
units = 512
out_timesteps = 12
img_in = (128, 128, 3)

def encode_sequences(tokenizer, seq_len, lines):
	sequence = tokenizer.texts_to_sequences(lines)
	sequence = pad_sequences(sequence, seq_len, padding="post")

	return sequence

xtrain, ytrain, xtest, ytest = train_test_split(images, captions, test_size=.2, random_state=12)

ytrain = encode_sequences(tokenizer, seq_len, ytrain)
ytest = encode_sequences(tokenizer, seq_len, ytest)

print (xtrain.shape, ytrain.shape)
print (xtest.shape, ytest.shape)

def get_encoder(insize):
	model = Sequential()
	model.add(Conv2D(128, (3,3), shape=insize, activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Conv2D(64, (3,3), activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Conv2D(32, (3,3), activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Flatten())

	model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

	return model

encoder = get_encoder(img_in)
encoder.summary()