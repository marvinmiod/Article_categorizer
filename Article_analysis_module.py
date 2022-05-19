# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:10:34 2022

Model to categorize unseen articles into 5 categories 
namely Sport, Tech, Business, Entertainment and Politics

@author: Marvin
"""

##%
import re
import pandas as pd
import numpy as np
import re
import datetime
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential #model is only for Sequential Model
from tensorflow.keras.layers import Dropout, Dense,  LSTM
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Saved path for the tokenizer
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')

#%%
## IMPORTANT!!#
# To use class module, must create empty python file name "__init__.py"
# and save in the same working folder as this file
# if not cannot call the module and will have error!

class ExploratoryDataAnalysis():
    def __init__(self): # reason why use __init__ to pass the data inside
        pass
    
    def remove_tags(self, data):
        """
        This function remove unwanted character and html tag from the text

        Parameters
        ----------
        data : Array
            Raw training data contains strings..

        Returns
        -------
        data : List/series
            Cleaned data in list..
        """
        for index, text in enumerate(data):
            data[index] = re.sub('<.*?>', ' ', text)
            
        return data
            
    def lower_split(self, data):
        """
        This function convert text into lower case and split into list
        Parameters
        ----------
        data : Array
            Raw training data contains strings.

        Returns
        -------
        data : List
            Cleaned data in list.
        """
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
     
        return data
    
    def text_tokenizer(self, data, token_save_path,
                           num_words=10000,oov_token='<OOV>', prt=False):
        """
        This function to assign number to each word and save it in a token file
        as dictionaries

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        token_save_path : TYPE
            DESCRIPTION.
        num_words : TYPE, optional
            DESCRIPTION. The default is 10000.
        oov_token : TYPE, optional
            DESCRIPTION. The default is '<OOV>'.
        prt : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        data : TYPE
            DESCRIPTION.
        """
         # OOV out of vocab
        
        # tokenizer to vectorize the words
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        
        # to save the tokenizer for deployment purposes
        
        token_json = tokenizer.to_json()
        
        with open(TOKENIZER_JSON_PATH, 'w') as json_file:
            json.dump(token_json, json_file)
        
        
        # to observe the number of words
        word_index = tokenizer.word_index
        
        if prt == True:
            print(word_index)
            print(dict(list(word_index.items())[0:10]))
        
        # to vectorize the sequence of text
            data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def text_pad_sequences(self, data):
        """
        Pad sequence is to ensure the data is in the same length eg. 200

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            Array int32 number with 2D shape and length/column 200.

        """
        return pad_sequences(data, maxlen=200,padding='post', 
                              truncating='post')
    
    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def lstm_layer(self,num_words, num_categories,
                   embedding_output=64, nodes=32, dropout_value=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout_value)) # dropout layer
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout_value))
        # output correspend to the y_test column shape 
        # y has to predict the review and the sentiment
        model.add(Dense(num_categories, activation='softmax')) 
        model.summary()
        plot_model(model)
        
        return model
        
    def simple_lstm_layer(self,num_words, num_categories,
                   embedding_output=64, nodes=32, dropout_value=0.2):
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout_value)) # dropout layer
        # output correspend to the y_test column shape
        # y has to predict the text and the category
        model.add(Dense(num_categories, activation='softmax')) 
        model.summary()
        plot_model(model)
        
        return model
    
    # can add def RNN_layer also
                                                      
class ModelEvaluation():
    def report_metrics(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        print('Accuracy score is', accuracy_score(y_true, y_pred))
      
        


#%%


URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(),'Log_article_analysis')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_article_analysis.h5')  

log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))


#%%


df = pd.read_csv(URL)
category = df['category']
text_article = df['text']
    
#%% EDA function call
# this is a test data that will be overwritten from
#  1) remove html tags from text, 2) lowercase and split each word 
# 3) tokenized it: assign number to each word 
# 4) add paddding sequence -> to ensure word is similar length

#eda = ExploratoryDataAnalysis() # call the function class for EDA
#test = eda.remove_tags(review) # remove html tag save the review text in to test var
#test = eda.lower_split(test) # convert text to lowercase and split

#test = eda.sentiment_tokenizer(test, token_save_path=TOKENIZER_JSON_PATH,
 #                             prt=True)

#test = eda.sentiment_pad_sequences(test)

#%% Call EDA function 
# Step 1: Data Cleaning

# assign eda variable to call function
eda = ExploratoryDataAnalysis()

tags_removed = eda.remove_tags(text_article) # remove html tag from review text

lower_split_text = eda.lower_split(tags_removed) # convert text to lowercase and split

#3) tokenized it: assign number to each word 
tokenized_text_article = eda.text_tokenizer(lower_split_text, 
                                            token_save_path=TOKENIZER_JSON_PATH, 
                                            prt=True)

# pad sequence to ensure the length of the text is the same length i.e: 200
pad_seq_text_article = eda.text_pad_sequences(tokenized_text_article)



#%% 
# Step 5) Data Pre-processing
# One Hot Encoding for Label (category)
one_hot_encoder = OneHotEncoder(sparse=False)
category_encoded = one_hot_encoder.fit_transform(np.expand_dims(category, 
                                                                 axis=-1))

#%%



# x is the article text that has been clean and pad sequenced, 
# y is the category
# split train test data 
x_train, x_test, y_train, y_test = train_test_split(pad_seq_text_article, 
                                                    category_encoded,
                                                    test_size=0.3,
                                                    random_state=123)

# remember to convert to 3 Dimension for the input shape for LSTM layer
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)



#%% test to inverse the combination of y_train:
# [1,0,0,0,0] business
# [0,1,0,0,0] entertainment
# [0,0,1,0,0] politics
# [0,0,0,1,0] sport
# [0,0,0,0,1] tech

print(y_train[6])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[6], axis=0)))


#%% Model creation function call

num_words = 10000

# to calculate number of total categories in the categories column
# business, entertainment, politics, sport, tech
num_categories = len(np.unique(category)) # categories there are 5
mc = ModelCreation()
model = mc.lstm_layer(num_words, num_categories)

#%% Compile & Model fitting

# tensorboard callback
tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)

# early stopping callback
early_stopping_callback = EarlyStopping(monitor='loss', patience=5 )

# compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics='accuracy')

# fit the x and y train data into the model and validate x/y test data
hist = model.fit(x_train, y_train, epochs=100, 
          validation_data=(x_test,y_test),
          callbacks=[tensorboard_callback, early_stopping_callback])

print(hist.history.keys())

#%% Model Evaluation

# preallocation of memory approach:
# number categories is auto-get from len(np.unique(category))
predicted_advanced = np.empty([len(x_test), num_categories])


for index, test in enumerate(x_test):
    predicted_advanced[index:] = model.predict(np.expand_dims(test, axis=0))
    
#%% Model analysis    

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

model_eval = ModelEvaluation()
model_eval.report_metrics(y_true, y_pred)

#%% Model Deployment
model.save(MODEL_SAVE_PATH)