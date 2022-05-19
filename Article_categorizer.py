# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:10:34 2022

Model to categorize unseen articles into 5 categories 
namely Sport, Tech, Business, Entertainment and Politics

@author: Marvin
"""




import pandas as pd
import numpy as np
import datetime
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from Article_analysis_module import ExploratoryDataAnalysis, ModelCreation
from Article_analysis_module import ModelEvaluation
from Article_analysis_module import TrainingHistory

#%% Saved path for static variable

URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'Saved_models', 
                                   'tokenizer_data.json')

LOG_PATH = os.path.join(os.getcwd(),'Log_article_analysis')

MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'Saved_models', 
                               'model_article_analysis.h5')  

log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))



#%% Step 1) Data Loading

df = pd.read_csv(URL)
category = df['category']
text_article = df['text']
    

#%% Call EDA function 
# Step 2) Data Cleaning

# assign eda variable to call function
eda = ExploratoryDataAnalysis()

# remove html tag from review text
tags_removed = eda.remove_tags(text_article) 

# convert text to lowercase and split
lower_split_text = eda.lower_split(tags_removed) 

#Step 3) tokenized it: assign number to each word 
tokenized_text_article = eda.text_tokenizer(lower_split_text, 
                                            token_save_path=TOKENIZER_JSON_PATH, 
                                            prt=True)
# to view the number of word per number
print(tokenized_text_article)

# pad sequence to ensure the length of the text is the same length i.e: 200
pad_seq_text_article = eda.text_pad_sequences(tokenized_text_article)



#%% 
# Step 4) Data Pre-processing
# One Hot Encoding for Label (category)
one_hot_encoder = OneHotEncoder(sparse=False)
category_encoded = one_hot_encoder.fit_transform(np.expand_dims(category, 
                                                                 axis=-1))

#%% Step 5) Prepare x/y training and test data

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

# number of words per tokenizer about 9779
num_words = 10000

# to auto-calculate number of total categories in the categories column
# business, entertainment, politics, sport, tech
num_categories = len(np.unique(category)) # categories there are 5
mc = ModelCreation()
model = mc.lstm_layer(num_words, num_categories)

#view model in console
plot_model(model)

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
hist = model.fit(x_train, y_train, epochs=50, 
          validation_data=(x_test,y_test),
          callbacks=[tensorboard_callback, early_stopping_callback])

print(hist.history.keys())

#%% Visualise the model using matplotlib

TrainingHistory.training_history(hist)

# to view the tensorboard go to browser type: http://localhost:6006/
# run in anaconda prompt tf_env: tensorboard --logdir "<path of log files>"

#%% Model Evaluation

# preallocation of memory approach:
# number categories is auto-get from len(np.unique(category))
predicted_advanced = np.empty([len(x_test), num_categories])


for index, test in enumerate(x_test):
    predicted_advanced[index:] = model.predict(np.expand_dims(test, axis=0))
    
#%% Model analysis    

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

# to view the Accuracy score, F1 score and confussion matrix plot
model_eval = ModelEvaluation()
model_eval.report_metrics(y_true, y_pred)


#%% Saved Model for Deployment
model.save(MODEL_SAVE_PATH)
