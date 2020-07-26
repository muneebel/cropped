
# coding: utf-8

# In[193]:



# Import libraries
import keras
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD
from keras.layers import BatchNormalization, Lambda, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.merge import Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf

import numpy as np

import glob
import os
from os import sys
import itertools

import cv2

import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[194]:


# Print library version

#python version

print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

#sklearn version
print('scikit-learn version is {}.'.format(sklearn.__version__))

#tensorflow version
print('tensorflow version is {}.'.format(tf.__version__))

#keras version
print('keras version is {}.'.format(keras.__version__))

#numpy version
print('numpy version is {}.'.format(np.__version__))

#opencv version
print('opencv version is {}.'.format(cv2.__version__))


# In[195]:


#define parameters

img_rows , img_cols = 227,227
num_classes = 3
batch_size = 32
nb_epoch = 2 # 5


# initialize label names for the dataset

labelNames = ["general_waste","green_sack", "mixed_recycling"]
classes = np.asarray([0, 1, 2])



# In[196]:


#create train test dataset

base_dir = '/'

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'test')

eval_dir = os.path.join(base_dir, 'eval')


# In[197]:


#create train dataset

trainX=np.zeros((1,227,227,3))
trainy=[]

paths=glob.glob(os.path.join('train/general_waste','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32) 
        im=im.reshape(1, 227, 227, 3)
        trainX=np.vstack((trainX, im))
        trainy.append(0)

print(str(trainX.shape))
trainX = np.delete(trainX, 0, axis=0)

print(str(trainX.shape))
#print(str(trainX))

paths=glob.glob(os.path.join('train/green_sack','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32) 
        im=im.reshape(1, 227, 227, 3)
        trainX=np.vstack((trainX, im))
        trainy.append(1)

paths=glob.glob(os.path.join('train/mixed_recycling','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32) 
        im=im.reshape(1, 227, 227, 3)
        trainX=np.vstack((trainX, im))
        trainy.append(2)


print('trainX.shape') 
print(str(trainX.shape))  

trainy = np.array(trainy)

print(str(trainy.shape)) 

trainy = np.expand_dims(trainy, axis=1)

print('trainy.shape') 
print(str(trainy.shape)) 


# In[198]:


# create validation data

testX=np.zeros((1,227,227,3))
testy=[]

paths=glob.glob(os.path.join('test/general_waste','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32) 
        im=im.reshape(1, 227, 227, 3)
        testX=np.vstack((testX, im))
        testy.append(0)


print(str(testX.shape))
#print(str(trainy.shape))
#print(str(testX))

testX = np.delete(testX, 0, axis=0)
print(str(testX.shape))


paths=glob.glob(os.path.join('test/green_sack','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32) 
        im=im.reshape(1, 227, 227, 3)
        testX=np.vstack((testX, im))
        testy.append(1)

print(str(testX.shape))


paths=glob.glob(os.path.join('test/mixed_recycling','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32)  
        im=im.reshape(1, 227, 227, 3)
        testX=np.vstack((testX, im))
        testy.append(2)


print(str(testX.shape))
#print(str(testX))

testy = np.array(testy)
testy = np.expand_dims(testy, axis=1)

print(str(testy.shape)) 


# In[199]:




# convert the labels from integers to vectors
lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)




# In[200]:


#define model for bin bags classification

def color_net(num_classes):
    # placeholder for input image
    input_image = Input(shape=(227,227,3))

    # ============================================= TOP BRANCH ===================================================
    # first top convolution layer

    top_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),
                              input_shape=(224,224,3),activation='relu')(input_image)


    top_conv1 = BatchNormalization()(top_conv1)
    top_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_conv1)

    # second top convolution layer
    # split feature map by half
    top_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(top_conv1)
    top_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(top_conv1)

    top_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv2)
    top_top_conv2 = BatchNormalization()(top_top_conv2)
    top_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv2)

    top_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv2)
    top_bot_conv2 = BatchNormalization()(top_bot_conv2)
    top_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv2)

    # third top convolution layer
    # concat 2 feature map
    top_conv3 = Concatenate()([top_top_conv2,top_bot_conv2])
    top_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_conv3)

    # fourth top convolution layer
    # split feature map by half
    top_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(top_conv3)
    top_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(top_conv3)

    top_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
    top_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)

    # fifth top convolution layer
    top_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
    top_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv5) 

    top_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)
    top_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv5)

    # ============================================= TOP BOTTOM ===================================================
    # first bottom convolution layer
    bottom_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),
                              input_shape=(227,227,3),activation='relu')(input_image)
    bottom_conv1 = BatchNormalization()(bottom_conv1)
    bottom_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_conv1)

    # second bottom convolution layer
    # split feature map by half
    bottom_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(bottom_conv1)
    bottom_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(bottom_conv1)

    bottom_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv2)
    bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
    bottom_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv2)

    bottom_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv2)
    bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
    bottom_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv2)

    # third bottom convolution layer
    # concat 2 feature map
    bottom_conv3 = Concatenate()([bottom_top_conv2,bottom_bot_conv2])
    bottom_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_conv3)

    # fourth bottom convolution layer
    # split feature map by half
    bottom_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(bottom_conv3)
    bottom_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(bottom_conv3)

    bottom_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
    bottom_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)

    # fifth bottom convolution layer
    bottom_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
    bottom_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv5) 

    bottom_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)
    bottom_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv5)

    # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
    conv_output = Concatenate()([top_top_conv5,top_bot_conv5,bottom_top_conv5,bottom_bot_conv5])

    # Flatten
    flatten = Flatten()(conv_output)

    # Fully-connected layer
    FC_1 = Dense(units=4096, activation='relu')(flatten)
    FC_1 = Dropout(0.6)(FC_1)
    FC_2 = Dense(units=4096, activation='relu')(FC_1)
    FC_2 = Dropout(0.6)(FC_2)
    output = Dense(units=num_classes, activation='softmax')(FC_2)
    
    model = Model(inputs=input_image,outputs=output)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    return model


# In[201]:



# initialise model
model = color_net(num_classes)

filepath = 'color_weights.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[202]:



# construct the image generator for data augmentation for training data
aug = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.3, horizontal_flip=True)



# In[203]:


# construct test set
testaug = ImageDataGenerator(rescale=1./255)


test_set = testaug.flow(
            x=testX, y=testy, 
            batch_size=batch_size)


# In[204]:


#train model

#steps_per_epoch=12000
#validation_steps=3000

steps_per_epoch=300
validation_steps=300


H = model.fit_generator(aug.flow(trainX, trainy, batch_size=batch_size),validation_data=test_set, epochs=nb_epoch,
         steps_per_epoch=steps_per_epoch, verbose=1,validation_steps=validation_steps,callbacks=callbacks_list)

# save model after training
model.save('color_model_final.h5')


# In[205]:


# plot the training loss and accuracy
i=0

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, nb_epoch+1), H.history["loss"],label="train_loss")
plt.plot(np.arange(1, nb_epoch+1), H.history["val_loss"],label="val_loss")
plt.plot(np.arange(1, nb_epoch+1), H.history["accuracy"],label="train_accuracy")
plt.plot(np.arange(1, nb_epoch+1), H.history["val_accuracy"],label="val_accuracy")
plt.title("Training Loss and Accuracy for model {}".format(i))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('fig.png')
plt.show()
#plt.close()


# In[206]:


# load model

model=load_model('color_model_final.h5')


# In[ ]:



# Prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm, classes, normalize=False, see_error=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        if see_error:
            np.fill_diagonal(cm, 0)
            title = "Error Analysis"
        else:
            title = "Normalized confusion matrix"
    else:
        title = "Confusion matrix, no normalization"
    print(cm)

    #plt.figure(figsize=(8,4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid(None)

    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, labelNames)
    plt.yticks(tick_marks, labelNames)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[ ]:


# classification report and confusion matrix for test data

testX = testX.astype("float") / 255.0

#print(testX)

predictions = model.predict(testX, batch_size=1)

print('Classification report')
report = classification_report(testy.argmax(axis=1),predictions.argmax(axis=1), target_names=labelNames)
print(report)

print('Confusion Matrix')
cm=confusion_matrix(testy.argmax(axis=1), predictions.argmax(axis=1))
print(cm)

plot_confusion_matrix(cm, classes, normalize=False, see_error=False, cmap=plt.cm.Blues)


# In[ ]:


# Evaluate model on unseen data
# create evaluation data

evalX=np.zeros((1,227,227,3))
evaly=[]

paths=glob.glob(os.path.join('eval/general_waste','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32) 
        im=im.reshape(1, 227, 227, 3)
        evalX=np.vstack((evalX, im))
        evaly.append(0)


print(str(evalX.shape))

evalX = np.delete(evalX, 0, axis=0)
print(str(evalX.shape))


paths=glob.glob(os.path.join('eval/green_sack','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32)  
        im=im.reshape(1, 227, 227, 3)
        evalX=np.vstack((evalX, im))
        evaly.append(1)

print(str(evalX.shape))


paths=glob.glob(os.path.join('eval/mixed_recycling','*.jpg'))


for path in paths:
        print(path)
        im=cv2.resize (cv2.imread(path), (227,227)).astype(np.float32)  
        im=im.reshape(1, 227, 227, 3)
        evalX=np.vstack((evalX, im))
        evaly.append(2)


print(str(evalX.shape))

evaly = np.array(evaly)
evaly = np.expand_dims(evaly, axis=1)

print(str(evaly.shape)) 

evalX = evalX.astype("float") / 255.0

#print(evalX)

# convert the labels from integers to vectors
lb = LabelBinarizer()
evaly = lb.fit_transform(evaly)








# In[ ]:


# classification report and confusion matrix for evaluation (unseen) data


predictions = model.predict(evalX, batch_size=1)


print('Classification report')
report = classification_report(evaly.argmax(axis=1),predictions.argmax(axis=1), target_names=labelNames)
print(report)

print('Confusion Matrix')

cm=confusion_matrix(evaly.argmax(axis=1), predictions.argmax(axis=1))
print(cm)

plot_confusion_matrix(cm, classes, normalize=False, see_error=False, cmap=plt.cm.Blues)

