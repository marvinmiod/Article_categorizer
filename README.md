# Article_categorizer
 Analyse article text and categorise Sport, Tech, Business, Entertainment and Politics.


# Description
This is a project to analyse article text from an URL sources and categorize the article under Sport, Tech, Business, Entertainment or Politics.

The dataset has been train using Sequential Neural Network model using Long Short Term Memory (LSTM) Bi-directional, embedding and dropout method with Early Stopping to prevent overfitting of data.


# How to use it
Clone the repo and run it.

Article_analysis_module.py is a script that contains the class module of the EDA, Model

Article_categorizer.py is a script for deployment of the test dataset

model_article_analysis.h5 is the saved model


# Outcome

The accuracy of the model is  using x hidden layers and x nodes.

The model accuracy is 92%

Summary Report and F1 score:

              precision    recall  f1-score   support

           0       0.98      0.92      0.95       156
           1       0.84      0.89      0.86       116
           2       0.92      0.94      0.93       121
           3       0.97      0.93      0.95       147
           4       0.89      0.93      0.91       128

    accuracy                           0.92       668
   macro avg       0.92      0.92      0.92       668
weighted avg       0.93      0.92      0.92       668


Suggestion 1: 
- to perform feature selection to reduce number of input into the training model

Suggestion 2: 
- to use Machine Learning pipeline to perform feature selection and re-train the dataset using Machine Learning model






# Credit
Credit goes to susanli2016 from GitHub for the dataset

https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv


# Images

Training model architecture
![image](image/model.png)


Training Epoch: loss and accuracy
![image](image/training_epoch_loss_accuracy.png)



Training process plotted using Tensorboard
![image](image/tensorboard.png)



Summary report and accuracy score
![image](image/Summary_report.png)


Training & validation accuracy plot
![image](image/training_accuracy.png)


Training & validation loss plot
![image](image/training_loss.png)

