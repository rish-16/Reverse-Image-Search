import string
import tensorflow as tf
import numpy as np
from numpy import array

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from preprocess import Extractor

ext = Extractor()		
ext.read_captions('../data/Flickr8k_text/Flickr8k.token.txt')
ext.read_images('../data/Flicker8k_Dataset/')

pairs = array(ext.find_pairs())

images = pairs[:, 1]
captions = pairs[:, 0]

captions = [s.translate((str.maketrans('', '', string.punctuation))) for s in captions]
captions = [captions[i].lower() for i in range(len(captions))]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

print ("Vocabulary: {}".format(len(tokenizer.word_index)))