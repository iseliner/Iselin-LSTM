# -*- coding: utf-8 -*-
"""
Made by iseliner.
Under construction, but testing for thesis is finished.
"""

##PREPROCESSING and LOADING OF DATA
import os
import gc
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#IMPORTS DATASET AND LABELS
print('Loading data...')

train_label = pd.read_csv('./data/data/labels/train/labels.train.csv')

dataset_path = ('./data/data/essays/train/original/')

bigram_essay_path = ('./features/ngram/essays/bigram/')
trigram_essay_path = ('./features/ngram/essays/trigram/')

speech_path = ('./data/data/speech_transcriptions/train/original/')

#           changes labels index to the test_taker_id
train_label = train_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
train_label = train_label.drop('speech_prompt', axis=1)
train_label = train_label.drop('essay_prompt', axis=1)

#Function which imports a datasets from a path and puts it into a list
def makeseq(path, listname):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read().split()
        read_file.close()
        listname.append(row)
    
#Slices elements that are too long and appends the shorter version
def slicefiles(target_df, vector_len):
    counter = 0
    for essay in target_df:
        if len(essay) > vector_len:
            old_new_sen = essay[0:vector_len]
            target_df[counter] = old_new_sen
        counter += 1
        
   
def padvectors(target_df, vector_len, embed_len):
    X = []
    for seq in target_df:
        sequence = [np.zeros(embed_len)] * vector_len
        for i,word in enumerate(seq):                               
            sequence[i] = word
        X.append(sequence)
    return X


def vectorizedata(target_df, target_model, embed_len):
    X = []
    for seq in target_df:
        sequence = []
        for word in seq:
            if word in target_model.wv.vocab:
                enc = target_model.wv[word]
                sequence.append(enc)
            else:
                enc = np.array([0]*embed_len)
                sequence.append(enc)
        X.append(sequence)
    return X    

def makelower(target_list):
    for x in range(len(target_list)):
        for y in range(len(target_list[x])):
            if target_list[x][y].isalpha():
                target_list[x][y] = target_list[x][y].lower()

#Make the label vectors: y_train(11000,11)
y = train_label.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_train = np_utils.to_categorical(encoded_y)                
                
#11000 elements, each containing all words in their respective essay

essay_vector_len = 350
essay_embed_len = 200
#Creates the dataset (!)
essay_df = []
makeseq(dataset_path, essay_df)
makelower(essay_df)

#sg=1 is skip-gram, cbow otherwise
print('Building Word2Vec...')

w2v = Word2Vec(sentences=essay_df, size=essay_embed_len, min_count=5, workers=6, window=5,sg=0)


print('Setting up x_train and y_train...')
slicefiles(essay_df, essay_vector_len)
essay_df = vectorizedata(essay_df, w2v, essay_embed_len)
X_train = padvectors(essay_df, essay_vector_len, essay_embed_len)
#X_train = np.array(X_train)

del essay_df
gc.collect()

#select_model = ExtraTreesClassifier()
#select_model.fit(X_train, y_train)
#X_train = select_model.transform(X_train)

X_train_essay = np.array(X_train)
X_train_essay = np.reshape(X_train_essay, (X_train_essay.shape))


#IVECTOR
print('Preparing data for input into the model...')
#Fetching i-vectors from distributed json file
ivector = []
with open('C:/Users/iseliner/Documents/programming/' +
          '/data/data/ivectors/train/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        ivector.append(data[x])

#ivector = np.array(ivector)
#clf = LinearDiscriminantAnalysis()
#clf.fit(ivector, encoded_y)
#X_new = clf.transform(ivector)
        
X_new = np.array(ivector)
 
X_train_ivec = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))


#SPEECH
#Creates the dataset (!)
speech_vector_len = 150
speech_embed_len = 100
speech_df = []
makeseq(speech_path, speech_df)
makelower(speech_df)

print('Building Word2Vec...')

w2v_speech = Word2Vec(sentences=speech_df, size=speech_embed_len, min_count=2, workers=6, window=5,sg=0)

slicefiles(speech_df, speech_vector_len)
speech_df = vectorizedata(speech_df, w2v_speech, speech_embed_len)
X_train_speech = padvectors(speech_df, speech_vector_len, speech_embed_len)
#X_train = np.array(X_train)

del speech_df
gc.collect()

#select_model = ExtraTreesClassifier()
#select_model.fit(X_train, y_train)
#X_train = select_model.transform(X_train)

X_train_speech = np.array(X_train_speech)
X_train_speech = np.reshape(X_train_speech, (X_train_speech.shape))

## MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
print('Building model...')
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

#Essay word input model
visible1 = Input(shape=(X_train_essay.shape[1], X_train_essay.shape[2]))
rnn11 = GRU(100, return_sequences=True, dropout=0.2)(visible1)
rnn12 = GRU(100, return_sequences=True, dropout=0.2)(rnn11)
rnn13 = GRU(100, return_sequences=False, dropout=0.2)(rnn12)
#rnn14 = GRU(200, dropout=0.2)(rnn13)
#rnn14 = GRU(100)(rnn13)
dense1 = Dense(30, activation='relu')(rnn13)


#Speech transcript word input model
visible2 = Input(shape=(X_train_speech.shape[1], X_train_speech.shape[2]))
rnn21 = GRU(60, return_sequences=True, dropout=0.2)(visible2)
rnn22 = GRU(60, dropout=0.2)(rnn21)
dense2 = Dense(20, activation='relu')(rnn22)

#i-vector input model
visible3 = Input(shape=(X_train_ivec.shape[1], 1))
rnn31 = GRU(60, return_sequences=False, dropout=0.2)(visible3)
#rnn32 = GRU(10, return_sequences=False)(rnn31)
#rnn33 = LSTM(10)(rnn32)
dense3 = Dense(10, activation='relu')(rnn31)

#Merge input-models
merge = concatenate([dense1,dense2,dense3])

#interpretation
#hidden1 =Dense(30)(lstm13)
output = Dense(11, activation='softmax')(merge)

model = Model(inputs=[visible1, visible2, visible2], outputs=output)

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop', 
              metrics=['accuracy'])

filepath = './saved_models/GRUensemble_essay350_essay3hidden100_150epoch_min5_cbow100_patience2_speech150_min2_cbow100_2hidden60_lowerspeec_loweressay.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min')
earlystop = EarlyStopping(patience=2, monitor='loss')
callbacks_list = [checkpoint, earlystop]

print(model.summary())
#plot_model(model,to_file='./LSTMensemble.png')

history = model.fit([X_train_essay, X_train_speech, X_train_ivec], y_train, epochs=150, batch_size=60, 
                       callbacks=callbacks_list)

score = model.evaluate([X_train_essay, X_train_speech, X_train_ivec], y_train, verbose=1)


##TEST
#////////////////////////////////////////////////////////////////////////
#TESTING ________________________________________________________________
test_label = pd.read_csv('./data/data/labels/dev/labels.dev.csv')
essay_test_path = ('./data/data/essays/dev/original/')
speech_test_path = ('./data/data/speech_transcriptions/dev/original/')

speech_test_path = ('./data/data/speech_transcriptions/dev/original/')

#           changes labels index to the test_taker_id
test_label = test_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
test_label = test_label.drop('speech_prompt', axis=1)
test_label = test_label.drop('essay_prompt', axis=1)

print('Setting labels')
y = test_label.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_test = np_utils.to_categorical(encoded_y)

#11000 elements, each containing all words in the essay
print('Initializing essay test data')
essay_test_df = []
makeseq(essay_test_path, essay_test_df)
slicefiles(essay_test_df, essay_vector_len)
makelower(essay_test_df)

essay_test_df = vectorizedata(essay_test_df, w2v, essay_embed_len)
essay_test_df = padvectors(essay_test_df, essay_vector_len, essay_embed_len)

X_test_essay = np.array(essay_test_df)
X_test_essay = np.reshape(X_test_essay, (X_test_essay.shape))

#IVEC
print('Preparing data for testing...')
#Fetching i-vectors from distributed json file
test_ivector = []
with open('C:/Users/iseliner/Documents/programming/' +
          '/data/data/ivectors/dev/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        test_ivector.append(data[x])

test_new = np.array(test_ivector)
#test_new = clf.transform(test_new)
X_test_ivec = np.reshape(test_new, (test_new.shape[0], test_new.shape[1], 1))

print('Initializing speech test data')
#SPEECH
speech_test_df = []
makeseq(speech_test_path, speech_test_df)
slicefiles(speech_test_df, speech_vector_len)
makelower(speech_test_df)

speech_test_df = vectorizedata(speech_test_df, w2v_speech, speech_embed_len)
speech_test_df = padvectors(speech_test_df, speech_vector_len, speech_embed_len)

X_test_speech = np.array(speech_test_df)
X_test_speech = np.reshape(X_test_speech, (X_test_speech.shape))

print('Running test set...')
predicted_L2 = model.evaluate([X_test_essay, X_test_speech, X_train_ivec], [y_test], batch_size=32)
print(predicted_L2)

#Prediction
prediction = model.predict([X_test_essay, X_test_speech, X_train_ivec], verbose=1)
print(prediction)

 
#
#Saves the history of the run
import matplotlib.pyplot as plt
import datetime
loglog = history.history
log_file = open('./logfile.txt', 'a')
log_file.write(str(datetime.datetime.now()) + '\n')
log_file.write(str(filepath) + '\n')
log_file.write('Training loss: ' + str(loglog['loss']) + '\n')
log_file.write('Training acc: ' + str(loglog['acc']) + '\n')
log_file.write('Test set: ' + str(predicted_L2) + '\n')
#log_file.write('F1: ' + str(f1) + '\n \n')


#RESULTS COUNTING. NOT FUNCTIONALITY FOR THE MODEL ___________________________
matrix_labels = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR',
                 'SPA', 'TEL', 'TUR']

predictionslist = []
x = 0
while x in range(len(prediction)):
    category_pred = np.argmax(prediction[x])
    predictionslist.append(matrix_labels[category_pred])
    x += 1
    
predictionslist_groundtruth = []
x = 0
while x in range(len(y_test)):
    category_pred = np.argmax(y_test[x])
    predictionslist_groundtruth.append(matrix_labels[category_pred])
    x += 1
    
from sklearn.metrics import f1_score
f1 = f1_score(predictionslist_groundtruth, predictionslist, average='macro')
print('F1:' + str(f1))
log_file.write('F1: ' + str(f1) + '\n \n')
log_file.close()

#MAKES CONFUSION MATRIX|||||||||||||||||||||||||||||||||||||||||||||||||||||||
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(predictionslist, predictionslist_groundtruth)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=matrix_labels,
                      title='Confusion matrix, without normalization')


