# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:17:58 2022

@author: AneeshDixit
"""

from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import model_from_json


from StoringRevs import Storing

rev_obj = Storing()

rev_df, rev_list = rev_obj.storingRevs()


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])


def remove_tags(string):
    # removelist = ""
    result = str(string)
    # result = ' '.join(ch for ch in string if(ch.isalnum() or ' '))    #remove non-alphanumeric characters
    # result = result.lower()
    return result


rev_df['review_text'] = rev_df['review_text'].apply(lambda cw: remove_tags(cw))
rev_df['review_text']

stop_words = set(stopwords.words('english'))

rev_df['review_text'] = rev_df['review_text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st


rev_df['review_text'] = rev_df.review_text.apply(lemmatize_text)

reviews = rev_df['review_text'].values

train_sentences, test_sentences = train_test_split(
    reviews, train_size=0.90)

vocab_size = 3000  # choose based on statistics
oov_tok = ''
embedding_dim = 100
max_length = 200  # choose based on statistics, for example 150 to 200
padding_type = 'post'
trunc_type = 'post'

# tokenize sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
word_index = tokenizer.word_index

# convert train dataset to sequence and pad sequences
train_sequences = np.array()

for arr in train_sentences:
    train_sequences.append(tokenizer.texts_to_sequences(arr))

#train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length)


# convert Test dataset to sequence and pad sequences
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)

prediction = loaded_model.predict(train_sentences)

pred_labels = []
for i in prediction:
    if i >= 0.8:
        pred_labels.append(5)
    elif i >= 0.6:
        pred_labels.append(4)
    elif i >= 0.4:
        pred_labels.append(3)
    elif i >= 0.2:
        pred_labels.append(2)
    elif i > 0:
        pred_labels.append(1)
    else:
        pred_labels.append(0)

print(pred_labels)
