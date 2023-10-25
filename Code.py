#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[4]:


import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2 as cv


# In[6]:


# replace this with your local directory
dir = '/Volumes/Transcend/Masters/Semester 1/Data Analytics and ML/Assignments/Assignment 3/DATASET'
get_ipython().system('ls')


# In[30]:


train_dir = '/Volumes/Transcend/Masters/Semester 1/Data Analytics and ML/Assignments/Assignment 3/DATASET/TRAIN'
test_dir = '/Volumes/Transcend/Masters/Semester 1/Data Analytics and ML/Assignments/Assignment 3/DATASET/TEST'


# In[31]:


# Directory with training organic pictures
train_organic_dir = os.path.join(train_dir, 'O')

# Directory with training recyclable pictures
train_recycle_dir = os.path.join(train_dir, 'R')


# In[32]:


print('total training organic images:', len(os.listdir(train_organic_dir)))
print('total training recycyle images:', len(os.listdir(train_recycle_dir)))


# In[33]:


# Directory with validation organic pictures
test_organic_dir = os.path.join(testing_dir, 'O')

# Directory with validation recyclable pictures
test_recycle_dir = os.path.join(testing_dir, 'R')


# In[34]:


print('total testing organic images:', len(os.listdir(test_organic_dir)))
print('total testing recycyle images:', len(os.listdir(test_recycle_dir)))


# In[35]:


train_organic_fnames = os.listdir(train_organic_dir)
train_recycle_fnames = os.listdir(train_recycle_dir)
train_recycle_fnames.sort()

get_ipython().run_line_magic('matplotlib', 'inline')

# Parameters for our graph; we'll output images in a 2x4 configuration
nrows = 2
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 2x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows *4)

pic_index += 4
next_organic_pix = [os.path.join(train_organic_dir, fname) 
                for fname in train_organic_fnames[pic_index-4:pic_index]]
next_recycle_pix = [os.path.join(train_recycle_dir, fname) 
                for fname in train_recycle_fnames[pic_index-4:pic_index]]

for i, img_path in enumerate(next_organic_pix + next_recycle_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)
  print(img.shape)

plt.show()
     
(188, 269, 3)
(164, 308, 3)
(197, 256, 3)
(225, 225, 3)
(160, 314, 3)
(150, 300, 3)
(187, 269, 3)
(168, 300, 3)


# In[36]:


from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


# In[37]:


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,               # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        shuffle=True,
        class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
testing_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        shuffle=True,
        class_mode='binary')


# In[38]:


# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + sigmoid output layer
model = Model(img_input, output)


# In[39]:


model.summary()


# In[40]:


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])


# In[41]:


history = model.fit(
      train_generator,
      epochs=8,
      validation_data=testing_generator,
      verbose=2,
      shuffle=True)


# In[42]:


#@title Graphing accuracy and loss
# Retrieve a list of accuracy results on training and testing data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and testing data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, label='training accuracy')
plt.plot(epochs, val_acc, label='testing accuracy')
plt.title('Training and testing accuracy')
plt.legend()

# Plot training and validation loss per epoch
plt.figure()
plt.plot(epochs, loss, label='training loss')
plt.plot(epochs, val_loss, label='testing loss')
plt.title('Training and testing loss')
plt.legend()


# In[43]:


# Flow validation images using val_datagen generator
test_visual = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        class_mode='binary',
        shuffle=True)

print(test_visual.class_indices)


# In[44]:


test_pred_prob = model.predict(test_visual)


# In[45]:


# must get index array before getting predictions!
test_dir_indices = test_visual.index_array
test_true_labels = [0 if n < 1112 else 1 for n in test_dir_indices] # directory is sorted alphanumerically; 1st 1112: 'O', 2nd 1112: 'R'

# getting predictions in the form of probablities 
test_pred_prob = model.predict(test_visual)

# converting the probablities into binary values 
test_pred_labels = [1 if n >= 0.5 else 0 for n in test_pred_prob]

print("Model predictions: "+str(test_pred_labels))
print("Actual labels:     "+str(test_true_labels))

# determining the filepaths of misclassified waste
num_misclasssified = 0
misclassified_filepaths = []
correctness = []
for pred_label, true_label, dir_index in zip(test_pred_labels, test_true_labels, test_visual.index_array):
  misclassified_filepaths.append(test_visual.filepaths[dir_index])
  if pred_label != true_label:
    correctness.append('incorrect')
    num_misclasssified += 1
  else:
    correctness.append('correct')

print("# of total images: "+str(len(correctness)))
print("# of misclassified images: "+str(num_misclasssified))


# In[47]:


# obtain images from the filepath at the determined indices
misclassified_imgs = []
for filepath in misclassified_filepaths:
  misclassified_imgs.append(mpimg.imread(filepath))

# plot first 30 images
f, axarr = plt.subplots(6,5, figsize=(20,10), constrained_layout=True)  # plt.subplots(row,cols)
count = 0
for r in range(6):
  for c in range(5):
    axarr[r,c].imshow(misclassified_imgs[count])
    if correctness[count] == 'correct':
      axarr[r,c].set_title(correctness[count])
    else:
      axarr[r,c].set_title(correctness[count], color='red')
    axarr[r,c].set_axis_off()
    count += 1 
plt.show()


# In[48]:


loss, acc = model.evaluate(testing_generator, verbose=1)
print("Accuracy using evaluate: "+str(acc))
print("Loss using evaluate: "+str(loss))


# In[49]:


# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# confusion matrix
matrix = confusion_matrix(test_true_labels, test_pred_labels, labels=[0, 1])
print('Confusion matrix : \n',matrix, '\n')

# outcome values order in sklearn
tp, fn, fp, tn = confusion_matrix(test_true_labels, test_pred_labels, labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn, '\n')

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(test_true_labels, test_pred_labels, labels=[1,0])
print('Classification report : \n',matrix)


# ## Saving the Model

# In[50]:


model.save('history.keras')


# In[52]:


import pickle
pickle.dump(history, open('model.pkl', 'wb'))


# In[53]:


model.save('my_model.h5') 


# In[ ]:





# In[ ]:




