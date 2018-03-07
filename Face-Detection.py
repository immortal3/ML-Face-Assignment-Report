#Importing Libraries
from keras.layers import Input,Flatten,Dense,Dropout
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from keras import optimizers
import numpy as np 
import pandas as pd
from keras import metrics
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from keras.preprocessing.image import load_img
from keras.preprocessing import image

#Providing seed - reapeated random generation
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(3)

#Loading data from preprocessing 
Data = np.load('NN_100_100_cropped.npy')
#images
X = Data[:,0]
#labels
Y = Data[:,1]

#Checking dimensions
X[0].shape

#Flattening the images
a = []
for i in X:
    a.append(i.flatten())
X_data = np.array(a)
print (X_data.shape)
X_data = X_data / 255

#Creating pandas frame for one-hot encoding using get_dummies function
df = pd.DataFrame(Y)
dummies = pd.get_dummies(df)
Y_data = np.array(pd.get_dummies(df))

#Checking dimensions after encoding
Y_data.shape

#Applying PCA by giving a number of features to be considered as input
PCA_output_n = 200
from sklearn.decomposition import PCA
pca = PCA(n_components=PCA_output_n)

#Fitting and transform
pca.fit(X_data)
print (pca.explained_variance_ratio_.sum())
X_temp_data = pca.transform(X_data)

#Test and Train Dataset Splitting
X_train, X_test, y_train, y_test = train_test_split(X_temp_data,Y, test_size=0.1)

#Modelling the SVM classifier
clf = svm.SVC(gamma=0.0001,C=5,kernel='rbf',degree=3,probability=True)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

#Checking things

import os
face_cascade = cv2.CascadeClassifier( os.getcwd() + '/haarcascade_frontalface_default.xml')
real_label = np.array(dummies.axes)

def predict_output(img_name):
    img = cv2.imread(os.getcwd()  + img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = np.array(faces)
    if faces.size:
        [x,y,w,h] = faces[0]
        cropped_img = img[y:y+h, x:x+w]
        cropped_img = cv2.resize(cropped_img,(100,100)).flatten()
        cropped_img = np.expand_dims(cropped_img,axis=0)
        cropped_img = cropped_img / 255
        features = pca.transform(cropped_img)
        x = clf.predict(features)
        print (img_name,x)

test_wild_images = os.listdir(os.getcwd() + '/test_in_wild/')
for i in test_wild_images:
    temp =predict_output('/test_in_wild/'+i)
    #print (i,' =>',temp)

# cross checking all photos
wrong_detect = []
def cross_verify():
    files = os.listdir(os.getcwd() + '/data/')
    data_img_and_label = []
    for f in files:
        temp_label = f.split('_')[0][:9]
        a = predict_output('/data/' +f)
        if a != temp_label:
            print ('Original :',f,'prediction:',a)
            wrong_detect.append(temp_label)

#cross_verify()