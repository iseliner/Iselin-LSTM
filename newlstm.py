# -*- coding: utf-8 -*-
"""
Made by iseliner
Still not finished and in the testing stages.
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
dataset_train_label = pd.read_csv('./data/labels/train/labels.train.csv')
#           changes labels index to the test_taker_id
dataset_label = dataset_train_label.set_index('test_taker_id')

#           drop the prompts, since we don't use them
dataset_label = dataset_label.drop('speech_prompt', axis=1)
dataset_label = dataset_label.drop('essay_prompt', axis=1)

dataset_path = ('./data/data/essays/train/original')

#Function which imports a datasets from a path and puts it into a list
def makeseq(path, listname):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read().split()
        read_file.close()
        listname.append(row)

#Makes vectors from te lists        
def makevectors(listname, vectorlist):
    for essay in listname:
        essay_sen = [np.zeros(100)] * 500
        for i,word in enumerate(essay):
            if word in model.wv.vocab:
                enc = model.wv[word]
                essay_sen[i] = enc
            else:
                model.build_vocab([word], update=True)
    vectorlist.append(essay_sen)

#11000 elements, each containing all words in their respective essay
label_list = []
x = 0
while x < len(train_label):
    label = train_label.iat[x,0]
    label_list.append(str(label))
    x += 1
    
df = []
makeseq(dataset_path, df)

counter = 0
for essay in df:
    if len(essay) > 500:
        new_essay = essay[500:len(essay)]
        old_new_sen = essay[0:500]
        label = label_list[counter]
        df[counter] = old_new_sen 
        df.append(new_essay)
        label_list.append(label)
    counter += 1


#Vocabulary (A list with all words in all essays as each individual list element)
#vocab = []
#for file in os.listdir(dataset_path):
#    read_file = open(dataset_path + str(file))
#    row = read_file.readline().split(' ')
#    while row != [''] :
#      for el in row :
#        vocab.append(el)
#      row = read_file.readline().split(' ')
#    read_file.close()
#vocab = df 

#Trains the world2vec model to vectorize data    
print('Initiate Word2Vec')
model = Word2Vec(size=100, min_count=0)
model.build_vocab(df)
#total_examples = model.corpus_count
print('Training Word2Vec')
model.train(df, total_examples=11000, epochs=30)


X = []
for essay in df:
    essay_sen = [np.zeros(100)] * 500
    for i,word in enumerate(essay):
      enc = model.wv[word]
      essay_sen[i] = enc
    X.append(essay_sen)

find_bigrams(X)
        

#Creates the TRAINING INPUT for the model  
x = np.array(X)

X_train = np.reshape(x, (x.shape))

label_train = pd.DataFrame(label_list)
#Make the label vectors: y_train(11000,11)
y = label_train.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_train = np_utils.to_categorical(encoded_y)

#MODEL _________________________________________________________________
print('Creating model...')

#Initalizing the RNN
nn_model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
nn_model.add(LSTM(200, return_sequences = True, 
               input_shape = (X_train.shape[1], X_train.shape[2])))
nn_model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
nn_model.add(LSTM(200, return_sequences = True))
nn_model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
nn_model.add(LSTM(200, return_sequences = True))
nn_model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
nn_nn_model.add(LSTM(200))

# Adding the output layer
nn_model.add(Dense(11, activation='softmax'))

#Should try the RMSPROP as optimizer
# Compiling the RNN
nn_model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'rmsprop', 
              metrics=['accuracy'])


# Fitting the RNN to the Training set
print('Fitting data to the model...')
nn_model.fit(X_train, y_train, epochs = 20, batch_size = 32)

#For later scoring
#score = model.evaluate(x_test, y_test, batch_size=16)

#////////////////////////////////////////////////////////////////////////
#TESTING ________________________________________________________________
test_label = pd.read_csv('./data/labels/train/labels.dev.csv')
test_path = ('./data/data/essays/dev/original')

#           changes labels index to the test_taker_id
test_label = test_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
test_label = test_label.drop('speech_prompt', axis=1)
test_label = test_label.drop('essay_prompt', axis=1)

y = test_label.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_test = np_utils.to_categorical(encoded_y)

#11000 elements, each containing all words in the essay
test_df = []
makeseq(test_path, test_df)


test = []
counter = 0
for essay in test_df:
    if len(essay) > 500:
        old_new_sen = essay[0:500]
        essay = old_new_sen
    essay_sen = [np.zeros(100)] * 500
    for i,word in enumerate(essay):
        if word in model.wv.vocab:
          enc = model.wv[word]
          essay_sen[i] = enc
        else:
            model.build_vocab([word], update=True)
    test.append(essay_sen)

x = np.array(test)
X_test = np.reshape(x, (x.shape))

predicted_L2 = nn_model.evaluate(X_test, y_test, batch_size=32)
