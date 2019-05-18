import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling3D
from keras.utils import np_utils,print_summary

sys.__stdout__ = sys.stdout
'''
Reading the data
'''
data=pd.read_csv("data.csv",',')
data=data.dropna()
dataset=np.array(data)
np.random.shuffle(dataset)

X=dataset[:]
Y=dataset[:]

image_x=31
image_y=33

X=X[:,1:]
Y=Y[:,0]

'''
Splitting the data into train and test
'''

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)




'''
one - hot encoding
'''



train_y=np_utils.to_categorical(Y_train)
test_y=np_utils.to_categorical(Y_test)

train_y=train_y.reshape(train_y.shape[0],train_y.shape[1])
test_y=test_y.reshape(test_y.shape[0],test_y.shape[1])
X_train=X_train.reshape(X_train.shape[0],image_x,image_y,1)
X_test=X_test.reshape(X_test.shape[0],image_x,image_y,1)

X_train=X_train/255
X_test=X_test/255
train_y=train_y/255
test_y=test_y/255


def keras_model(image_x,image_y):
    num_of_classes=10
    model=Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(31,33,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

    model.add(Conv2D(64,(5,5),input_shape=(31,33,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    
    model.add(Conv2D(64,(5,5),activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    
    model.add(Flatten())
    model.add(Dense(10,activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1,activation='softmax'))

    model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['accuracy'])
    filepath='DevanagriScript.h5'
    checkpoint1=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list=[checkpoint1]

    return model,callbacks_list

model,callbacks_list=keras_model(image_x,image_y)
model.fit(X_train,train_y,validation_data=(X_test,test_y),epochs=10,callbacks=callbacks_list)
scores=model.evaluate(X_test,test_y,verbose=0)
print("Error : ",(100-scores[1]*100))
print_summary(model)

model.save('DevanagriScript.h5')
