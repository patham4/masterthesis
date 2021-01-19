#!/usr/bin/env python
# coding: utf-8


from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import np_utils
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot


import os
import array
import pydicom
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import time
import sklearn
from sklearn.model_selection import train_test_split
import keras
#from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import Deconvolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
# load vgg model
from keras.applications.vgg16 import VGG16
# load the model
# plot feature map of first conv layer for given image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pickle
from keras.models import model_from_json





# =============================================
# Grad-CAM code
# =============================================
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from keras import activations


###############################################
#Title: Basic Classification: Prefict a clothing item
#Author: Tensorflow- François Chollet
#Date: 2017
#Availability: https://www.tensorflow.org/tutorials/keras/classification

#MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

f = open('exp6/objs.pkl', 'rb')
obj = pickle.load(f)
f.close()




num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
Blab=np.ones(len(obj))
Blab = Blab.astype(int)
n_classes=2




json_file = open('modeldiscriminatorT1T2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("modelweightsT1T2.h5")

loaded_model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

predictions=loaded_model.predict(obj)

test_loss, test_acc = loaded_model.evaluate(obj,  Blab, verbose=2)
print('\nTest accuracy:', test_acc)





def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(n_classes))
    plt.yticks([])
    #print(predictions_array.shape)
    thisplot = plt.bar(range(n_classes), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')




plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    #print(i,predictions[i].shape,Blab.shape,B.shape)
    plot_image(i, predictions[i], Blab, obj[i,:,:,0])
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    #print(i,predictions[i,:].shape,Blab.shape)
    plot_value_array(i, predictions[i], Blab)
plt.tight_layout()
plt.show()


#########################################################




###############################################
#Title: Visualizing Keras CNN attention: Grad-CAM Class Activation Maps
#Author: Chris
#Date: 2019
#Availability: https://www.machinecurve.com/index.php/2019/11/28/visualizing-keras-cnn-attention-grad-cam-class-activation-maps/



# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(loaded_model, 'visualized_layer')
# Swap softmax with linear
loaded_model.layers[layer_index].activation = activations.linear
loaded_model = utils.apply_modifications(loaded_model) 
# Numbers to visualize
indices_to_visualize = list(range(0, 22))




# Visualize
for index_to_visualize in indices_to_visualize:
    # Get input
    input_image = obj[index_to_visualize]
    input_class = int(Blab[index_to_visualize])
    # Matplotlib preparations
    fig, axes = plt.subplots(1, 3)
    # Generate visualization
    visualization = visualize_cam(loaded_model, layer_index, filter_indices=input_class, seed_input=input_image)
    axes[0].imshow(input_image[..., 0], cmap='gray') 
    axes[0].set_title('Input')
    axes[1].imshow(visualization)
    axes[1].set_title('Grad-CAM')
    heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
    original = np.uint8(cm.gray(input_image[..., 0])[..., :3] * 255)
    axes[2].imshow(overlay(heatmap, original))
    axes[2].set_title('Overlay')
    fig.suptitle(f'Brain of scanner = {input_class}')
    plt.show()

#######################################################################



