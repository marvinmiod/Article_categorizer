# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:10:34 2022

Model to categorize unseen articles into 5 categories 
namely Sport, Tech, Business, Entertainment and Politics

@author: Marvin
"""

#%%

import re
import os
import pandas as pd
import numpy as np
#from Sentiment_analysis_module import ExploratoryDataAnalysis, ModelCreation
#from Sentiment_analysis_module import ModelEvaluation
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(),'Log_article_analysis')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_article_analysis.h5')  



#%%


# EDA
# Step 1) Data Loading
df = pd.read_csv(URL)
category = df['category']
text = df['text']



#%%
# Step 2) Data cleaning, remove tag 
# assign different variable name
eda = ExploratoryDataAnalysis()
tags_removed = eda.remove_tags(review) # remove html tag from review text
lower_split_text = eda.lower_split(tags_removed) # convert text to lowercase and split

# step 3) Feature selection
# step 4) Data Vectorization
# Assign each word with number
token_text_review = eda.sentiment_tokenizer(lower_split_text, 
                                            token_save_path=TOKENIZER_JSON_PATH, 
                                            prt=True)
# pad the word to ensure same length=200
pad_seq_review = eda.sentiment_pad_sequences(token_text_review)



# Step 5) Data Pre-processing
# One Hot Encoding for Label (y test and train)
one_hot_encoder = OneHotEncoder(sparse=False)
sentiment_encoded = one_hot_encoder.fit_transform(np.expand_dims(sentiment, 
                                                                 axis=-1))

# to calculate number of total categories
num_categories = len(np.unique(sentiment))

# x = review, y = sentiment
# train test split
x_train, x_test, y_train, y_test = train_test_split(pad_seq_review, 
                                                    sentiment_encoded,
                                                    test_size=0.3,
                                                    random_state=123)

# Convert x_train and x_test to 3 Dimension for the input shape
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# to inverse the number to positive and negative review
# to check [0,1] is positive and [1,0] is negative
print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0)))
