#....
Author : Sahithi kodali
....#

#Import required libraries and packages into the coding environment

import cv2
import os
from glob import glob 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout
import sklearn 
from sklearn.model_selection import train_test_split
 
#Code to extract frames from each video

folder_names = os.listdir('/content/drive/DDD Project/DDD')
path = '/content/drive/DDD Project/frames_data'
for i in [0,5,10]:
    print(i)
    for folder_name in folder_names:
        print(folder_name)
        vidcap = cv2.VideoCapture('/content/drive/DDD Project/DDD'+'/'+folder_name+'/'+str(i)+'.mp4')
        success,image = vidcap.read()
        count = 0
        if success==False:
            print('check '+ folder_name)
        while success:
            if count%100==0:
                cv2.imwrite(path+'/'+str(i)+'/'+'frame'+str(i)+'_'+str(folder_name)+'_%d.jpg' % count, image)    
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1

#Code for finding the shape of the images
 
import cv2
import os
path = '/content/drive/DDD Project/frames_data/'
image_shapes = []
for i in [0,5,10]:
    print(i)
    folder_names = ['01','02','04','06','07','08','09','10','14','16','17','18','19','21','22','25','27','28','30']
    for folder in folder_names:
        print(folder)
        img = cv2.imread(path+str(i)+'/'+'frame'+str(i)+'_'+folder+'_100.jpg')
        shape = img.shape
        if shape not in image_shapes:
            image_shapes.append(shape)
            print(image_shapes)
print(image_shapes)

image_shapes = [(480, 720, 3), (720, 1280, 3), (2560, 1440, 3), (1080, 1920, 3), (1920, 1080, 3), (1280, 720, 3), (352, 640, 3)]
img = cv2.imread('/content/drive/DDD Project/frames_data/5/frame5_30_1000.jpg')
sorted(glob('/content/drive/DDD Project/frames_data/'+str(0)+'/frame'+str(0)+'_'+'04'+'_*.jpg'))

#Code for rotating and removing non-essential frames in the images
 
path = '/content/drive/DDD Project/frames_data/'
for i in [0]:
    folders = ['01']
    for f in folders:
        for im in sorted(glob(path+str(i)+'/frame'+str(i)+'_'+f+'_*.jpg')):
            img = cv2.imread(im)
            img_rotated = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(im,img_rotated)

#resizing the image
folder_names = ['01','02','04','06','07','08','09','10','14','16','17','18','19','21','22','25','27','28','30']
path = '/content/drive/DDD Project/frames_data/'
for i in [0,5,10]:
    for folder in folder_names:
        images_path = sorted(glob(path+str(i)+'/frame'+str(i)+'_'+folder+'_*.jpg'))
        for image_path in images_path:
            img=cv2.imread(image_path)
            shape = img.shape
            if shape == (480,270,3):
                img =cv2.resize(img,(135,240),interpolation = cv2.INTER_AREA)
                cv2.imwrite(image_path,img)

folders = ['01','02','04','06','07','08','09','10','14','16','17','18','19','21','22','25','27','28','30']
total_folders = os.listdir('/content/drive/DDD Project/DDD')
for i in total_folders:
    if i not in folders:
        for k in [0,5,10]:
            rem_imgs = sorted(glob('/content/drive/DDD Project/frames_data/'+str(k)+'/'+'frame'+str(k)+'_'+i+'_*.jpg'))
            if len(rem_imgs)!=0:
                for p in rem_imgs:
                    os.remove(p)          

#Code for building the model of CNN architecture

model = tf.keras.Sequential()
model.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer = 'he_uniform',padding='same',input_shape = (240,135,3)))
model.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer = 'he_uniform',padding='same'))
model.add(MaxPooling2D(2,2)) 
model.add(Conv2D(128,(3,3),activation = 'relu',kernel_initializer = 'he_uniform',padding='same'))
model.add(Conv2D(128,(3,3),activation = 'relu',kernel_initializer = 'he_uniform',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(512,(3,3),activation = 'relu',kernel_initializer = 'he_uniform',padding='same'))
model.add(Conv2D(512,(3,3),activation = 'relu',kernel_initializer = 'he_uniform',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512,activation = 'relu',kernel_initializer = 'he_uniform'))
model.add(Dense(128,activation = 'relu',kernel_initializer = 'he_uniform'))
model.add(Dense(3,activation = 'softmax'))


#Provide the ground truth for the model and save the image numpy arrays of the images

zero = sorted(glob('/content/drive/DDD Project/frames_data/0/*.jpg'))
five = sorted(glob('/content/drive/DDD Project/frames_data/5/*.jpg'))
ten = sorted(glob('/content/drive/DDD Project/frames_data/10/*.jpg'))
ground_truth = np.zeros((len(zero)+len(five)+len(ten),))
ground_truth[0:len(zero),] = 0
ground_truth[len(zero):+len(zero)+len(five),] = 1
ground_truth[len(zero)+len(five):len(zero)+len(five)+len(ten),] = 2
 
def data_reading(image_paths):
    data = np.zeros((len(image_paths),240,135,3))
    for i in range(len(image_paths)):
        print(i)
        data[i,:,:,:] = cv2.imread(image_paths[i])
    return data.astype('float32')/255
 
x_train = data_reading(x_train)
np.save('/content/drive/DDD Project/frames_data/x_train.npy',x_train)
x_test = data_reading(x_test)
np.save('/content/drive/DDD Project/frames_data/x_test.npy',x_test)
np.save('/content/drive/DDD Project/frames_data/y_test.npy',y_test)
np.save('/content/drive/DDD Project/frames_data/y_train.npy',y_train)

#Loading data from the numpy arrays and training the model
 
x_train = np.load('/content/drive/DDD Project/frames_data/x_train.npy')
x_test = np.load('/content/drive/DDD Project/frames_data/x_test.npy')
y_train = np.load('/content/drive/DDD Project/frames_data/y_train.npy')
y_test = np.load('/content/drive/DDD Project/frames_data/y_test.npy')

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Test the model and evaluate the Accuracy

model.fit(x_train,y_train,batch_size=50,epochs=5,verbose=1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
