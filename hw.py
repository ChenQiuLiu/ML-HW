#加载包

import os
import glob
import random
import numpy as np
import pandas as pd

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

from PIL import Image

from tensorflow.keras.utils import to_categorical

import seaborn as sns
import matplotlib.image as img
import matplotlib.pyplot as plt




#数据集加载
train_csv = pd.read_csv("./Training_set.csv") #图片名称，label
test_csv = pd.read_csv("./Testing_set.csv")   #图片名称
train_fol = glob.glob("./train/*") 
test_fol = glob.glob("./test/*")


#train_csv

#train_csv.label.value_counts()



#展示类别分布

#import plotly.express as px
#l = train_csv.label.value_counts()
#fig = px.pie(train_csv, values=l.values, names=l.index, title='Distribution of Human Activity')
#fig.show()


filename = train_csv['filename'] #图片名字
situation = train_csv['label']   #标签



#处理数据
img_data = []
img_label = []
length = len(train_fol)
for i in (range(len(train_fol)-1)):
    t = '../input/human-action-recognition-har-dataset/Human Action Recognition/train/' + filename[i]    
    temp_img = Image.open(t)
    img_data.append(np.asarray(temp_img.resize((160,160))))  
    img_label.append(situation[i])
    
inp_shape = (160, 160,3)
iD = img_data
iD = np.asarray(iD)
#type(iD)
    


y_train = to_categorical(np.asarray(train_csv['label'].factorize()[0]))
#print(y_train[0])



#模型构建
vgg_model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,  
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet') #使用预训练模型

for layer in pretrained_model.layers:
        layer.trainable=False #冻结base layer


vgg_model.add(pretrained_model)

#待会训练以下层的参数
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))

#使用 adam优化器
vgg_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

#vgg_model.summary()



history = vgg_model.fit(iD,y_train, epochs=60)



#保留model
vgg_model.save_weights("model_hw.h5")


#查看loss
losss = history.history['loss']
plt.plot(losss)

#查看accu
accu = history.history['accuracy']
plt.plot(accu)




#测试


# Function to read images as array
def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((160,160)))
    
    
# Function to predict

def test_predict(test_image):
    result = vgg_model.predict(np.asarray([read_image(test_image)]))

    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]
    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)

    image = img.imread(test_image)
    plt.imshow(image)
    plt.title(prediction)
    


test_predict('./test/Image_101.jpg')
test_predict('./test/Image_1050.jpg')