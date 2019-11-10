import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Embedding, Flatten
from tensorflow.keras.text.preprocessing import pad_sequences