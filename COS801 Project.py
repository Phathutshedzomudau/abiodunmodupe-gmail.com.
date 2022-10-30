#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install h5py


# In[2]:


pip install keras


# In[3]:


pip install scikit-learn


# In[4]:


pip install numpy scipy


# In[5]:


pip install pillow


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import os
from tab2img.converter import Tab2Img
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models
import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization
import graphviz # for plotting decision tree graphs
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# The first dataset we are going to start with Home Equity dataset (HMEQ) contains baseline and loan performance information for 5,960 recent home equity loans. The target (BAD) is a binary variable indicating whether an applicant eventually defaulted or was seriously delinquent. This adverse outcome occurred in 1,189 cases (20%). For each applicant, 12 input variables were recorded. Loading the Data Set

# In[7]:


#path = 'C:/Users/PHATHUTSHEDZO/Downloads/IGTD-main/IGTD-main/Data/'
path = 'C:/Users/PHATHUTSHEDZO/Downloads/COS801/project/IGTD-main/IGTD-main/Data'
#filename = 'hmeq.csv'
filename = 'australian.csv'
file = os.path.join(path,filename)
dataset = pd.read_csv(file,
                   names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13", "A14", "A15"] )


# In[ ]:





# In[8]:


dataset.head()


# In[9]:


#dataset=dataset.iloc[:, 1:] GERMANY
#dataset.info()


# In[10]:


#dataset['Risk'] = dataset['Risk'].map({'bad':1, 'good':0}) GERMANY


# In[11]:


#dataset['Saving accounts'] = dataset['Saving accounts'].fillna('Others')
#dataset['Checking account'] = dataset['Checking account'].fillna('Others') GERMANY


# numpy array to one hot encoding

# In[12]:


OHE = LabelEncoder()
##one_hot_encoded_data = ['JOB','REASON']
##dataset[one_hot_encoded_data] = dataset[one_hot_encoded_data].apply(OHE.fit_transform)
##one_hot_encoded_dataset = ['Sex', 'Housing', 'Credit amount', 'Saving accounts','Risk','Checking account','Purpose']
###dataset[one_hot_encoded_dataset] = dataset[one_hot_encoded_dataset].apply(OHE.fit_transform) 
# Replace the null features with 0:
dataset.fillna(0, inplace=True) # Re-check N/A was replaced with 0.
X = pd.DataFrame(dataset.iloc[:, 0:14].values)
y = dataset.iloc[:, 14].values
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 4)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
X_test


# In[13]:


X


# In[14]:


y


# In[ ]:





# In[15]:


dataset


# In[ ]:





# In[ ]:





# In[16]:


y


# Spliting dataset

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


def fitting(X1, y1, criterion, splitter, mdepth, clweight, minleaf):

    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit the model
    model = tree.DecisionTreeClassifier(criterion=criterion, 
                                        splitter=splitter, 
                                        max_depth=mdepth,
                                        class_weight=clweight,
                                        min_samples_leaf=minleaf, 
                                        random_state=0, 
                                  )
    clf = model.fit(X_train, y_train)

    # Predict class labels on training data
    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

    # Tree summary and model evaluation metrics
    print('*************** Tree Summary ***************')
    print('Classes: ', clf.classes_)
    print('Tree Depth: ', clf.tree_.max_depth)
    print('No. of leaves: ', clf.tree_.n_leaves)
    print('No. of features: ', clf.n_features_in_)
    print('--------------------------------------------------------')
    print("")
    
    print('*************** Evaluation on Test Data ***************')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')
    print("")
    
    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')
    
    # Use graphviz to plot the tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X.columns, 
                                class_names=[str(list(clf.classes_)[0]), str(list(clf.classes_)[1])],
                                filled=True, 
                                rounded=True, 
                                #rotate=True,
                               ) 
    graph = graphviz.Source(dot_data)
    
    # Return relevant data for chart plotting
    return X_train, X_test, y_train, y_test, clf, graph

# Select data for modeling
X1=dataset[['A2','A3','A4']]
y1=dataset['A15'].values

# Fit the model and display results
X_train, X_test, y_train, y_test, clf, graph = fitting(X1, y1, 'gini', 'best', 
                                                       mdepth=3, 
                                                       clweight=None,
                                                       minleaf=1000)

# Plot the tree graph
##graph

# Save tree graph to a PDF
#graph.render('Decision_Tree_all_vars_gini')

ANN1 = Sequential()
ANN1.add(Dense(60, activation='relu', kernel_initializer='uniform',input_dim=14))
ANN1.add(Dense(6, activation='relu', kernel_initializer='uniform'))
ANN1.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
ANN1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.fit(X_train, y_train, batch_size=32, epochs=10)
ANN1.fit(X_train, y_train, batch_size = 100, epochs = 100)
yANN1_pred = ANN1.predict(X_test)
yANN1_pred = (yANN1_pred > 0.5)
yANN1_pred
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, yANN1_pred)
print(cm)
accuracy_score(y_test,yANN1_pred)
print('Precision: %.3f' % precision_score(y_test, yANN1_pred))
print('Recall: %.3f' % recall_score(y_test, yANN1_pred))
print('F1 Score: %.3f' % f1_score(y_test, yANN1_pred))


# define model
DSNN1 = Sequential()
DSNN1.add(Dense(1, input_shape=(14,), activation='relu'))
DSNN1.add(Dense(1, activation='sigmoid'))
# compile model
DSNN1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = DSNN1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)
yDSNN1_pred = DSNN1.predict(X_test)
yDSNN1_pred = (yDSNN1_pred < 0.5)
yDSNN1_pred
print('Precision: %.3f' % precision_score(y_test, yDSNN1_pred))
print('Recall: %.3f' % recall_score(y_test, yDSNN1_pred))
print('F1 Score: %.3f' % f1_score(y_test, yDSNN1_pred))

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn1 = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn1.fit(X_train, y_train)

#Predict the response for test dataset
yknn1_pred = knn1.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, yknn1_pred))
print('Precision: %.3f' % precision_score(y_test, yknn1_pred))
print('Recall: %.3f' % recall_score(y_test, yknn1_pred))
print('F1 Score: %.3f' % f1_score(y_test, yknn1_pred))

from sklearn.svm import SVC
# Building and fit the classifier
SMV1 = SVC(kernel='rbf', gamma=0.01, C=1000)
SMV1.fit(X_train, y_train)
# Make predictions and check the accuracy
ysmv1_pred = SMV1.predict(X_test)
print(accuracy_score(y_test, ysmv1_pred))
print('Precision: %.3f' % precision_score(y_test, ysmv1_pred))
print('Recall: %.3f' % recall_score(y_test, ysmv1_pred))
print('F1 Score: %.3f' % f1_score(y_test, ysmv1_pred))

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
NB1 = GaussianNB()

#Train the model using the training sets
NB1.fit(X_train, y_train)

#Predict the response for test dataset
yNB1_pred = NB1.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, yNB1_pred))

print('Precision: %.3f' % precision_score(y_test, yNB1_pred))
print('Recall: %.3f' % recall_score(y_test, yNB1_pred))
print('F1 Score: %.3f' % f1_score(y_test, yNB1_pred))

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Conv1D
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, BatchNormalization, MaxPooling1D, LeakyReLU
from sklearn.model_selection import train_test_split
#dataset = pd.read_csv('CAP.data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

ms_input_shape = (None,1000,9) #Could someone suggest how I should set the input shape to be?
X_train = np.random.random((10,1000,9))
y_train = np.random.random((10, 5))

CNN1 = keras.models.Sequential()
#model.add(tf.keras.layers.InputLayer())
CNN1.add(Conv1D(filters=6, kernel_size=21, strides=1, padding='same', activation='relu',input_shape= ms_input_shape[1:],kernel_initializer=keras.initializers.he_normal()))
CNN1.add(BatchNormalization()) #what is the purpose of this!
CNN1.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
CNN1.add(Conv1D(filters=16, kernel_size=5, strides=1, padding='same',activation='relu'))
CNN1.add(BatchNormalization())
CNN1.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
CNN1.add(Flatten())
CNN1.add(Dense(120, activation='relu'))
CNN1.add(Dense(84))
CNN1.add(Dense(5, activation='softmax'))
CNN1.summary()

import tensorflow as tf
CNN1.compile(optimizer=tf.keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=['acc'])


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)]

BATCH_SIZE = 32 #I am not sure what batch size should I set for my case?
EPOCHS = 20

history = CNN1.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=callbacks_list,
                    validation_split=0.2,
                    verbose=1)


# In[ ]:





# In[18]:


path = 'C:/Users/PHATHUTSHEDZO/Downloads/COS801/project/IGTD-main/IGTD-main/Data'
filename = 'german_credit_data.csv'
file = os.path.join(path,filename)
data = pd.read_csv(file)
data
#dataset=pd.read_csv('german_credit_data.csv')
#print("The dataset is {} credit record".format(len(dataset)))


# In[19]:


OHE = LabelEncoder()
one_hot_encoded_data = ['Sex','Housing','Saving accounts','Checking account','Purpose','Risk']
data[one_hot_encoded_data] = data[one_hot_encoded_data].apply(OHE.fit_transform)
# Replace the null features with 0:
data.fillna(0, inplace=True) # Re-check N/A was replaced with 0.


# In[20]:


X = pd.DataFrame(data.iloc[:, 0:10].values)
y = data.iloc[:, 10].values
X


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[22]:


x_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[23]:


# Initialising the RNN
model = Sequential()
#Adding the First input hidden layer and the LSTM layer
# return_sequences = True, means the output of every time step to be shared with hidden next layer
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding the Second hidden layer and the LSTM layer
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# Compiling the RNN
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32)
model.save('credit_prediction.h5')

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)
y_pred

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test,y_pred)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


# All solution for different dataset were found using this code 

# In[ ]:




