# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# skip-gramm model
# REPO: https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb

import os

from core import (
    models,
    files
)

FeaturesFolder = 'input-DMKD-300'
ContextFolder = 'DATASET' #TIME

current_folder = os.path.dirname(os.path.abspath(__file__))
print('current_folder:',current_folder)

# load features
input_files = files.load_list(
        path=current_folder,
        folder=FeaturesFolder #config['IO']['InputFolder']
    )
files_list=input_files

#input_file = current_folder + '/' +  InputFolder
print('input_files:', input_files)

def read_tokens(input_file):
        return list(
            files.read_file(
                filename=input_file.path
            )
        )

# read tokens
for file in files_list:
        input_tokens = read_tokens(
            input_file=file
        )

print('input_tokens:', input_tokens)

###########################
###### load contexts #######
###########################
for token in input_tokens:
        print('token:', token)

# load features
input_context_data = files.load_list(
        path=current_folder,
        folder=ContextFolder #config['IO']['InputFolder']
    )
print("input_context_data: ",input_context_data)
input_context_files = input_context_data

def find_contexts(input_file, token):
        return files.search_contexts_in_file(
                filename=input_file.path,token=token
            )


# read contexts
formed_token_contexts = []
for token in input_tokens:#[:1]
        #print('token:', token)
        for file in input_context_files:
                #print('file:', token)
                input_token_contexts = find_contexts(
                    input_file=file, token=token
                )
                if not input_token_contexts.contexts:
                    print("no context: ", input_token_contexts)
                else:
                    print("has context: ", input_token_contexts)
                    formed_token_contexts.append(input_token_contexts)
        #print("input_token_contexts: ",input_token_contexts)

print("formed_token_contexts: ",formed_token_contexts)

# ==== Start  skip-gramm-softmax.py =====

# here we have word set by which we will have word vector
# all words from contexts to 1 big array
words = []
for token in formed_token_contexts:
    for text in token.contexts:
        #print('######## text ############', text)
        for word in text.split(' '):
            words.append(word)

words = set(words) # our sample
print('######## words ############', words)

# ===== Data generation =====
# Display word and its int - need to understand how to forming this in
word2int = {}

for i,word in enumerate(words):
    #print("\n i", i, "word: ", word)
    word2int[word] = i # indexed words by i
print("######## word2int ########", word2int)

# Forming sentences of word arrays
sentences = []
for token in formed_token_contexts:
    for context in token.contexts:
       #for sentence in corpus:
       sentences.append(context.split())
print('######## sentences ############', sentences)

# define window size
WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        #print("idx: ", idx, " word: ", word)
        #TODO: could we skip 2-3 most nearest and check next after them
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] :
            if neighbor != word:
                data.append([word, neighbor])


import pandas as pd
# for text in sentences:
#     print(text)

df = pd.DataFrame(data, columns = ['input', 'label'])
print(df.head(len(data))) # 10 get data.size
print("shape: ", df.shape)



# ===== Define Tensorflow Graph =====
import tensorflow as tf
import numpy as np

# Build INPUT layer (one hot vector) - embedding
ONE_HOT_DIM = len(words)

# function to convert numbers to one hot vectors
# encoding vectors like [0 1 0 0 0 0 0 ],
# ONE_HOT_DIM is dimenstion of vector
# data_point_index is the position of vector
def to_one_hot_encoding(data_point_index):
    try:
        one_hot_encoding = np.zeros(ONE_HOT_DIM)
        one_hot_encoding[data_point_index] = 1
        #print("\n one_hot_encoding: ", one_hot_encoding)
        return one_hot_encoding
    except Exception as e:
        print("Exception to_one_hot_encoding: ",e)
        return -1

# build encodedvectors of words and their neighbours
X = [] # input word
Y = [] # target word

for x, y in zip(df['input'], df['label']):
    #print("\n x: ", x, " y: ", y)
    x_word2int = -1
    y_word2int = -1

    try:
        x_word2int = word2int[ x ]
    except Exception as e:
        print("Exception word2int: ",e)

    try:
        y_word2int = word2int[ y ]
    except Exception as e:
        print("Exception word2int: ",e)

    if x_word2int != -1 and y_word2int != -1:
        x_to_one_hot_encoding = word2int[ x ]
        y_to_one_hot_encoding = word2int[ y ]
        if x_to_one_hot_encoding != -1 and y_to_one_hot_encoding != -1:
            X.append(to_one_hot_encoding(word2int[ x ])) # word
            Y.append(to_one_hot_encoding(word2int[ y ])) # word neigbour

# print("\n X: ", X)
# print("\n Y: ", Y)

# convert them to numpy arrays
X_train = np.asarray(X)
Y_train = np.asarray(Y)

print("\n X_train: ", X_train)
print("\n Y_train: ", Y_train)

# making placeholders for X_train and Y_train
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

# Build HIDDEN layer (linear neuron)
# word embedding will be 2 dimension for 2d visualization
EMBEDDING_DIM = 2

# hidden layer: which represents word vector eventually
# https://en.wikipedia.org/wiki/Cross_entropy - very common to cost function
# https://www.youtube.com/watch?v=tRsSi_sqXjI
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1])) #bias -  systematic error. https://en.wikipedia.org/wiki/Bias
hidden_layer = tf.add(tf.matmul(x,W1), b1)
print("hidden_layer W1: ", W1)
print("hidden_layer b1: ", b1)
print("hidden_layer: ", hidden_layer)

# Build OUTPUT layer (softmax)
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

# loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)



## ===== Train =====
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 20000
for i in range(iteration):
    # input is X_train which is one hot encoded word
    # label is Y_train which is one hot encoded neighbor word
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))

#TODO: save to model

# Now the hidden layer (W1 + b1) is actually the word look up table
vectors = sess.run(W1 + b1)
print("vectors: ", vectors)

# word vector in table
w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]
pd.set_option('display.max_rows', 1000)
print(w2v_df)

###########
###########
import math
p1 = vectors[9]
p2 = vectors[9]
# formula
# distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

word_index = input('Enter index of word you would like to find the nearest: ')
print('Index', word_index)
IND = int(word_index) #9
P = vectors[IND] #features[IND] #
print(">>>> P >>>>> ", P)
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=20)
knn.fit(vectors)


# NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=5, p=2,radius=1.0, warn_on_equidistant=True)
closest = knn.kneighbors([P], return_distance=True)
print("closest_node: ", closest)
indexes = closest[1][0]
distances = closest[0][0]
print("indexes: ", indexes)
print("distances: ", distances)
# for close in closest:
#     print("close: ", close)


# TODO: code-optimization just to display the word and index
lookingForWord = "undefined"
lookingForIndex = -1
for index,word in enumerate(words):
    if index == IND:
        lookingForWord = word
        lookingForIndex = index
        # print("Looking nearest for: index: ", index, "word: ", word)


# Retreive features
features = []
for token in input_tokens:
        print('token:', token.term)
        features.append(files.replace_term_phrase_with_dash(token.term))
print('features:', features)

# DISPLAY nearby 10
print("  ################## Nearest ################## ")
print("nearest TO index: ", lookingForIndex, " : ", lookingForWord)
print(" ")
for i,value in enumerate(indexes):
    for index,word in enumerate(words):
        if index == value and lookingForIndex != value and word in features:
            print("index: ", index, "word: ", word, "distance: ", distances[i])

# could be code-optimization
# print(list(words.keys())[list(words.values()).index(value)])
# print("word: ", words[value])


# TODO: filter out the words
for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    distance = math.sqrt(((p1[0]-x1)**2)+((p1[1]-x2)**2) )
    # print("p1[0]: ", p1[0], "p1[1]: ",p1[1])
    # print("x1: ", x1, "x2: ",x2)
    print("word: ", word, "distance: ",distance)
    print("\n")

# word vector in Chart
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    if word in features:
        ax.annotate(word, (x1,x2 ))

PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

plt.xlim(x_axis_min,x_axis_max)
plt.ylim(y_axis_min,y_axis_max)
plt.rcParams["figure.figsize"] = (10,10)

plt.show()
