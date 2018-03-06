# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:48:09 2018

@author: Blue
"""

#Import the libraries
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence

import gensim
from gensim.models import Word2Vec



#IMPORTS DATASET AND LABELS
print('Loading data...')
'''
dataset_train_label = pd.read_csv('./data/labels/train/labels.train.csv')
#           changes labels index to the test_taker_id
dataset_label = dataset_train_label.set_index('test_taker_id')

#           drop the prompts, since we don't use them
dataset_label = dataset_label.drop('speech_prompt', axis=1)
dataset_label = dataset_label.drop('essay_prompt', axis=1)

#Make the label vectors: y_train(11000,11)
y = dataset_label.values

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_train = encoded_y#np_utils.to_categorical(encoded_y)

'''

#dataset_path = ('./data/essays/original/')
dataset_path = ('./data/essays/tokenized/')


#Prepare the input for the model
df = []
for file in os.listdir(dataset_path):
    read_file = open(dataset_path + str(file))
    row = read_file.readline().split()
    read_file.close()
    df.append(row)
    
df_essay = pd.DataFrame(df).values


#Vocabulary
vocab = []
for file in os.listdir(dataset_path):
    read_file = open(dataset_path + str(file))
    row = read_file.readline().split(' ')
    while row != [''] :
      for el in row :
        vocab.append(el)
      row = read_file.readline().split(' ')
    read_file.close()
    

#Word2Vec
sentences = []

for file in os.listdir(dataset_path):
    read_file = open(dataset_path + str(file))
    row = read_file.readline().split('\n')
    while row != [''] :
      sentences.append(row[0])
      row = read_file.readline().split('\n')
    read_file.close()
    
sentences_2 = pd.DataFrame(sentences).values

    
model = Word2Vec(size=200, min_count=0)
model.build_vocab(vocab)
total_examples = model.corpus_count
model.train(sentences, total_examples=11000, epochs=30)

model.wv

voc_vec = gensim.models.word2vec.Word2Vec(vocab, min_count=1)
print(voc_vec)

raise


#Build vocabulary
X = []
for file in df:
    vector = vectorizer.fit(file)
    X.append(vector)

X_train = np.reshape(x, (x.shape[0], x.shape[1], 1))


#MODEL _________________________________________________________________
print('Creating model...')

#Initalizing the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, 
               input_shape = (X_train.shape[1], 1)))
#model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
#model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
#model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
#model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 11))

#Should try the RMSPROP as optimizer
# Compiling the RNN
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam', 
              metrics=['accuracy'])



#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||| TRAINING start ||||||||||||||||||||||||||||||||||||||
# Fitting the RNN to the Training set
print('Fitting data to the model...')
model.fit(X_train, y_train, epochs = 50, batch_size = 1)









































