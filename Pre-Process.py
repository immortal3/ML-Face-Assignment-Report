#Importing necessary libraries

import os
from keras.preprocessing.image import load_img
import numpy as np
from keras.preprocessing import image
import numpy as np
import cv2


#Fetching the data

files = os.listdir(os.getcwd() + '/data/')
face_cascade = cv2.CascadeClassifier( os.getcwd() + '/haarcascade_frontalface_default.xml')

data_img_and_label = []

#--------------------------------------------------------------------------------------------------------
#Optional - VGG face implementation - Import Libraries
# from keras_vggface.vggface import VGGFace
# from keras.engine import  Model
# from keras.layers import Input

#vgg_features = VGGFace(include_top=False, input_shape=(80, 80, 3), pooling='avg')
#--------------------------------------------------------------------------------------------------------

#Flattening and cropping
import time
i = 0
for f in files:
	img = cv2.imread(os.getcwd() + '/data/' + f)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	faces = np.array(faces)
	if faces.size:
		[x,y,w,h] = faces[0]
		cropped_img = img[y:y+h, x:x+w]
		cropped_img = cv2.resize(cropped_img,(100,100))
		temp_label = f.split('_')[0][:9]
		#cropped_img = np.expand_dims(cropped_img,axis=0)
		#features = vgg_features.predict(cropped_img).flatten()		
		data_img_and_label.append((cropped_img,temp_label))
		print (temp_label,cropped_img.shape)
	i = i + 1

#Converting to .npy file
data_img_and_label = np.array(data_img_and_label)
np.save('NN_100_100_cropped.npy',data_img_and_label)
