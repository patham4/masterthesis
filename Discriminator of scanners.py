#!/usr/bin/env python
# coding: utf-8


from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import np_utils
from keras.utils.np_utils import to_categorical

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
from numpy import asarray


import imgaug
import imgaug.augmenters as iaa
from mpl_toolkits.axes_grid1.inset_locator import inset_axes




# =============================================
# Grad-CAM code
# =============================================
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from keras import activations



#LOAD REGISTERED IMAGES

###############################################
#Title: Kaggle - Data Science Bowl 2017
#Author: Wrosinski
#Date: 2017
#Availability: https://www.kaggle.com/chenyanan/the-details

def load_scan(path,machine):
    #print(os.listdir(path))
    a=[s for s in os.listdir(path) if not s.startswith('.')]
    a=sorted([x for x in a])
    #print(a)
    slices = [pydicom.dcmread(path + "/" + s,force="True") for s in a]
    image = np.stack([s.pixel_array for s in slices])
    return image


# In[ ]:
############################################################

#For registered images
def load_data():
    #INPUT_FOLDER='/media/data/penriquez/Images_256'
    #INPUT_FOLDER='/Users/arturo/Documents/MATLAB/THESIS/Images_256_unregistered'
    INPUT_FOLDER='/Users/arturo/Desktop/registered_full_dataset'
    #INPUT_FOLDER='/Users/arturo/Desktop/Images_test 2'
    patients=[f for f in os.listdir(INPUT_FOLDER) if not f.startswith('.')]
    machinename1='Optima'
    machinename2='SIGNA'
    machine1=[]
    machine2=[]
    for i in patients:
        INPUT_FOLDER2=os.path.join(INPUT_FOLDER, i)
        scanners=[f for f in os.listdir(INPUT_FOLDER2) if not f.startswith('.')]
        #print(scanners)
        j=scanners[0]
        m=scanners[1]
        sequences3=[]
        INPUT_FOLDER30=os.path.join(INPUT_FOLDER2, j)
        INPUT_FOLDER31=os.path.join(INPUT_FOLDER2, m)
            
        sequences1=[m for m in os.listdir(INPUT_FOLDER30) if not m.startswith('.')]
        sequences1.sort
        x=array.array('i',(1 for i in range(4,5)))
        sequences2=[a for a in sequences1 if str.find(a,"T1Mapping")>=0]
            
            
        sequences3=[m for m in os.listdir(INPUT_FOLDER31) if not m.startswith('.')]
        sequences3.sort
        x=array.array('i',(1 for i in range(4,5)))
        sequences4=[a for a in sequences3 if str.find(a,"T1Mapping")>=0]
            
            #print(sequences2)
            #print(sequences2)
            #hola=[b for b in sequences if str.find(b,"T2Map")>=0]
            #sequences3=sequences2+hola
        for k,l in zip(sequences2,sequences4):
            INPUT_FOLDER41=os.path.join(INPUT_FOLDER30, k)
            INPUT_FOLDER42=os.path.join(INPUT_FOLDER31, l)
                #print(INPUT_FOLDER4)
            images1=[m for m in os.listdir(INPUT_FOLDER41) if not m.startswith('.')]
            images1.sort
                
            images2=[m for m in os.listdir(INPUT_FOLDER41) if not m.startswith('.')]
            images2.sort
            #print(images1,images2)
                #print(images)
            if (i==patients[0]) and (k==sequences2[0]) and (l==sequences4[0]) :
                if machinename1 in images1[0]:
                    #print('Optima 1 is in the first folder')
                    first_patient_pixels1=load_scan(INPUT_FOLDER41,machinename1)
                    machine1=np.array(first_patient_pixels1)
                    first_patient_pixels2=load_scan(INPUT_FOLDER42,machinename2)
                    machine2=np.array(first_patient_pixels2)
                        #print('here')
                        #print(machine1)
                elif machinename1 in images2[0]:
                        #print('optima 1 is in the second folder')
                        first_patient_pixels1=load_scan(INPUT_FOLDER42,machinename1)
                        machine1=np.array(first_patient_pixels1)
                        first_patient_pixels2=load_scan(INPUT_FOLDER41,machinename2)
                        machine2=np.array(first_patient_pixels2)
                    
                    
                elif machinename2 in images1[0]:
                                #print('Sigma 1 is in the first folder')
                                first_patient_pixels1=load_scan(INPUT_FOLDER42,machinename1)
                                machine1=np.array(first_patient_pixels1)
                                first_patient_pixels2=load_scan(INPUT_FOLDER41,machinename2)
                                machine2=np.array(first_patient_pixels2)
                    
                elif machinename2 in images2[0]:
                    #print('sigma 1 is in the second folder')
                    first_patient_pixels1=load_scan(INPUT_FOLDER41,machinename1)
                    machine1=np.array(first_patient_pixels1)
                    first_patient_pixels2=load_scan(INPUT_FOLDER42,machinename2)
                    machine2=np.array(first_patient_pixels2)
                        #print(machine2)
            else:
                if machinename1 in images1[0]:
                    if not len(machine1):
                        #print('optima in first folder 1')
                        first_patient_pixels1=load_scan(INPUT_FOLDER41,machinename1)
                        machine1=np.array(first_patient_pixels1)
                        first_patient_pixels2=load_scan(INPUT_FOLDER42,machinename2)
                        machine2=np.concatenate([np.array(machine2), np.array(first_patient_pixels2)])
                        
                            #print(machine1)
                    else:
                        #print('optima in first folder 2')
                        first_patient_pixels1=load_scan(INPUT_FOLDER41,machinename1)
                        machine1 = np.concatenate([np.array(machine1), np.array(first_patient_pixels1)])
                        first_patient_pixels2=load_scan(INPUT_FOLDER42,machinename2)
                        machine2 = np.concatenate([np.array(machine2), np.array(first_patient_pixels2)])
                            #print(machine1)
                elif machinename1 in images2[0]:
                    if not len(machine1):
                        #print('optima in second folder 1')
                        first_patient_pixels1=load_scan(INPUT_FOLDER42,machinename1)
                        machine1=np.array(first_patient_pixels1)
                        first_patient_pixels2=load_scan(INPUT_FOLDER41,machinename2)
                        machine2 = np.concatenate([np.array(machine2), np.array(first_patient_pixels2)])
                            #print(machine1)
                    else:
                        #print('optima in second folder 2')
                        first_patient_pixels1=load_scan(INPUT_FOLDER42,machinename1)
                        machine1 = np.concatenate([np.array(machine1), np.array(first_patient_pixels1)])
                        first_patient_pixels2=load_scan(INPUT_FOLDER41,machinename2)
                        machine2 = np.concatenate([np.array(machine2), np.array(first_patient_pixels2)])
                if machinename2 in images1[0]:
                    if not len(machine2):
                        #print('sigma in first folder 1')
                        first_patient_pixels2=load_scan(INPUT_FOLDER41,machinename2)
                        machine2=np.array(first_patient_pixels2)
                        first_patient_pixels1=load_scan(INPUT_FOLDER42,machinename1)
                        machine1=np.concatenate([np.array(machine1), np.array(first_patient_pixels1)])
                        
                            #print(machine1)
                    else:
                        #print('sigma in first folder 2')
                        first_patient_pixels2=load_scan(INPUT_FOLDER41,machinename2)
                        machine2 = np.concatenate([np.array(machine2), np.array(first_patient_pixels2)])
                        first_patient_pixels1=load_scan(INPUT_FOLDER42,machinename1)
                        machine1 = np.concatenate([np.array(machine1), np.array(first_patient_pixels1)])
                    
                            #print(machine1)
                elif machinename2 in images2[0]:
                    if not len(machine2):
                        #print('sigma in second folder 1')
                        first_patient_pixels2=load_scan(INPUT_FOLDER42,machinename2)
                        machine2=np.array(first_patient_pixels2)
                        first_patient_pixels1=load_scan(INPUT_FOLDER41,machinename1)
                        machine1=np.array(first_patient_pixels1)
                            #print(machine1)
                    else:
                        #print('sigme in second folder 2')
                        first_patient_pixels2=load_scan(INPUT_FOLDER42,machinename2)
                        machine2 = np.concatenate([np.array(machine2), np.array(first_patient_pixels2)])
                        first_patient_pixels1=load_scan(INPUT_FOLDER41,machinename1)
                        machine1 = np.concatenate([np.array(machine1), np.array(first_patient_pixels1)])

                        
    machine1=expand_dims(machine1, axis=-1)
    #print(machine1.size)
    machine2=expand_dims(machine2, axis=-1)
    #print(machine2.size)
    #print(machine1)
    #print(machine2)
    totalpatients=np.concatenate([machine1,machine2])
    print(totalpatients.shape)
    #labelszero=np.zeros([269]) #When we load only the training set
    #labelsone=np.ones([269])
    labelszero=np.zeros([323]) #When we load the full dataset
    labelsone=np.ones([323])
    
    #labelszero=np.zeros([54])
    #labelsone=np.ones([54])
    totallabels=np.concatenate([labelszero,labelsone])
    return(asarray(totalpatients),totallabels)




#LOAD UNREGISTERED IMAGES


###############################################
#Title: Kaggle - Data Science Bowl 2017
#Author: Wrosinski
#Date: 2017
#Availability: https://www.kaggle.com/chenyanan/the-details

def load_scan(path):
    #slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices = [pydicom.dcmread(path + "/" + s,force="True") for s in os.listdir(path)]
    slices = [s for s in slices if "SliceLocation" in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    #print(slices.shape)
    machine=np.array([s.ManufacturerModelName for s in slices])
    lab=[]
    #lab=np.empty([2,2])
    for s in slices:
        if str.find(s.ManufacturerModelName,"Optima")>=0:
            lab.append(0)
        else:
            lab.append(1)
    #lab=[0 for s in slices if str.find(s.ManufacturerModelName,"Optima")>0 & 1 for s in slices if str.find(s.ManufacturerModelName,"Optima")<0]
    #lab.append(1 for s in slices if str.find(s.ManufacturerModelName,"Optima")<=0)

    #print(np.array(lab))
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    #print(image.shape)
    image = image.astype(np.int16)
    #print(machine)
    dictOfWords = { i : image for i in machine }
    return np.array(image, dtype=np.int16), lab, dictOfWords
    #return dictOfWords

###########################################################


def load_data1():
    #INPUT_FOLDER='/media/data/penriquez/Images_256_unregistered'
    INPUT_FOLDER='/Users/arturo/Documents/MATLAB/THESIS/Images_256_unregistered'
    #INPUT_FOLDER='/Users/arturo/Documents/MATLAB/THESIS/Images_test'
    patients=[f for f in os.listdir(INPUT_FOLDER) if not f.startswith('.')]
    machine1=[]
    machine2=[]
    for i in patients:
        INPUT_FOLDER2=os.path.join(INPUT_FOLDER, i)
        scanners=[f for f in os.listdir(INPUT_FOLDER2) if not f.startswith('.')]
        for j in scanners:
            sequences3=[]
            INPUT_FOLDER3=os.path.join(INPUT_FOLDER2, j)
            sequences=[m for m in os.listdir(INPUT_FOLDER3) if not m.startswith('.')]
            sequences.sort
            #TENGO QUE REDUCIR SEQUENCES A SOLO LAS QUE YO QUIERA
            x=array.array('i',(1 for i in range(4,5)))
            sequences2=[a for a in sequences if str.find(a,"Mapping")>=0]
            #ONLY FOR RUN 6
            sequences3=[a for a in sequences if str.find(a,"T2MapPreset-1")>=0]
            #REST OF RUNS
            #sequences3=[a for a in sequences if str.find(a,"Mapping")>=0]
            #print(sequences2,sequences3)
            #print(sequences2)
            #hola=[b for b in sequences if str.find(b,"T2Map")>=0]
            #sequences3=sequences2+hola
            for k,m in zip(sequences2,sequences3):
                INPUT_FOLDER40=os.path.join(INPUT_FOLDER3, k)
                INPUT_FOLDER41=os.path.join(INPUT_FOLDER3, m)
                patient0 = load_scan(INPUT_FOLDER40)
                patient1 = load_scan(INPUT_FOLDER41)
                first_patient_pixels00, machine00, datas00 = get_pixels_hu(patient0)
                first_patient_pixels01, machine01, datas01 = get_pixels_hu(patient1)
                #print(machine00,machine01)
                if (k==sequences2[0]) & (j==scanners[0]) & (i==patients[0]):
                    #totalpatients=np.array(first_patient_pixels)
                    #shortlabels=np.array(machine)
                    #print(totallabels)
                    if np.any(machine00)==1:
                        machine1=np.array(first_patient_pixels00)
                    if np.any(machine01)==0:
                        #print('here')
                        machine2=np.array(first_patient_pixels01)
                else:
                    #totalpatients = np.concatenate([np.array(totalpatients), np.array(first_patient_pixels)])
                    #shortlabels = np.concatenate([np.array(shortlabels), np.array(machine)])
                    if np.any(machine00)==1:
                        if not len(machine1):
                            machine1=np.array(first_patient_pixels00)
                        else:
                            machine1 = np.concatenate([np.array(machine1), np.array(first_patient_pixels00)])
                    if np.any(machine01)==0:
                        if not len(machine2):
                            #print('or here')
                            machine2=np.array(first_patient_pixels01)
                        else:
                            #print('oooor here')
                            machine2=np.concatenate([np.array(machine2), np.array(first_patient_pixels01)])



#totallabels = np.concatenate([np.array(totallabels), np.array(machine)])
                #plt.imshow(first_patient_pixels[10], cmap=plt.cm.gray)
                #plt.show()
                #np.size(machine)
                #print(first_patient_pixels.shape)
                #totalpatients.append(first_patient_pixels)
                #totallabels.append(machine)
                #print(totallabels)
    #pd.DataFrame.to_csv('images.csv',totalpatients,shortlabels,totallabelsbig)
    machine1=expand_dims(machine1, axis=-1)
    machine2=expand_dims(machine2, axis=-1)
    totalpatients=np.concatenate([machine1,machine2])
    #aug=augment_images(machine1)
    #machine1augmented=np.concatenate([machine1,aug])
    #machine2augmented=np.concatenate([machine2,machine2])
    #labelszero=np.zeros([323]) #When we load only the training set
    #labelsone=np.ones([323])
    labelszero=np.zeros([269]) #When we load only the training set
    labelsone=np.ones([269])
    totallabels=np.concatenate([labelszero,labelsone])
    return(asarray(totalpatients),asarray(totallabels))




#LOAD T1vsT2 IMAGES

###############################################
#Title: Kaggle - Data Science Bowl 2017
#Author: Wrosinski
#Date: 2017
#Availability: https://www.kaggle.com/chenyanan/the-details

def load_scan(path):
    #slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices = [pydicom.dcmread(path + "/" + s,force="True") for s in os.listdir(path)]
    slices = [s for s in slices if "SliceLocation" in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    #print(slices.shape)
    machine=np.array([s.ManufacturerModelName for s in slices])
    lab=[]
    #lab=np.empty([2,2])
    for s in slices:
        if str.find(s.ManufacturerModelName,"Optima")>=0:
            lab.append(0)
        else:
            lab.append(1)
    #lab=[0 for s in slices if str.find(s.ManufacturerModelName,"Optima")>0 & 1 for s in slices if str.find(s.ManufacturerModelName,"Optima")<0]
    #lab.append(1 for s in slices if str.find(s.ManufacturerModelName,"Optima")<=0)

    #print(np.array(lab))
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    #print(image.shape)
    image = image.astype(np.int16)
    #print(machine)
    dictOfWords = { i : image for i in machine }
    return np.array(image, dtype=np.int16), lab, dictOfWords
    #return dictOfWords
###############################################################

# In[16]:
def load_data1():
    #INPUT_FOLDER='/media/data/penriquez/Images_256_unregistered'
    INPUT_FOLDER='/Users/arturo/Documents/MATLAB/THESIS/Images_256_unregistered'
    patients=[f for f in os.listdir(INPUT_FOLDER) if not f.startswith('.')]
    machine1=[]
    machine2=[]
    for i in patients:
        INPUT_FOLDER2=os.path.join(INPUT_FOLDER, i)
        scanners=[f for f in os.listdir(INPUT_FOLDER2) if not f.startswith('.')]
        for j in scanners:
            sequences3=[]
            INPUT_FOLDER3=os.path.join(INPUT_FOLDER2, j)
            sequences=[m for m in os.listdir(INPUT_FOLDER3) if not m.startswith('.')]
            sequences.sort
            #TENGO QUE REDUCIR SEQUENCES A SOLO LAS QUE YO QUIERA
            x=array.array('i',(1 for i in range(4,5)))
            sequences2=[a for a in sequences if str.find(a,"Mapping")>=0]
            sequences3=[a for a in sequences if str.find(a,"T2MapPreset-1")>=0]
            #print(sequences2,sequences3)
            #print(sequences2)
            #hola=[b for b in sequences if str.find(b,"T2Map")>=0]
            #sequences3=sequences2+hola
            for k,m in zip(sequences2,sequences3):
                INPUT_FOLDER40=os.path.join(INPUT_FOLDER3, k)
                INPUT_FOLDER41=os.path.join(INPUT_FOLDER3, m)
                patient0 = load_scan(INPUT_FOLDER40)
                patient1 = load_scan(INPUT_FOLDER41)
                first_patient_pixels00, machine00, datas00 = get_pixels_hu(patient0)
                first_patient_pixels01, machine01, datas01 = get_pixels_hu(patient1)
                #print(machine00,machine01)
                if (k==sequences2[0]) & (j==scanners[0]) & (i==patients[0]):
                    #totalpatients=np.array(first_patient_pixels)
                    #shortlabels=np.array(machine)
                    #print(totallabels)
                    if np.any(machine00)==1:
                        machine1=np.array(first_patient_pixels00)
                    if np.any(machine01)==1:
                        #print('here')
                        machine2=np.array(first_patient_pixels01)
                else:
                    #totalpatients = np.concatenate([np.array(totalpatients), np.array(first_patient_pixels)])
                    #shortlabels = np.concatenate([np.array(shortlabels), np.array(machine)])
                    if np.any(machine00)==1:
                        if not len(machine1):
                            machine1=np.array(first_patient_pixels00)
                        else:
                            machine1 = np.concatenate([np.array(machine1), np.array(first_patient_pixels00)])
                    if np.any(machine01)==1:
                        if not len(machine2):
                            #print('or here')
                            machine2=np.array(first_patient_pixels01)
                        else:
                            #print('oooor here')
                            machine2=np.concatenate([np.array(machine2), np.array(first_patient_pixels01)])



#totallabels = np.concatenate([np.array(totallabels), np.array(machine)])
                #plt.imshow(first_patient_pixels[10], cmap=plt.cm.gray)
                #plt.show()
                #np.size(machine)
                #print(first_patient_pixels.shape)
                #totalpatients.append(first_patient_pixels)
                #totallabels.append(machine)
                #print(totallabels)
    #pd.DataFrame.to_csv('images.csv',totalpatients,shortlabels,totallabelsbig)
    machine1=expand_dims(machine1, axis=-1)
    machine2=expand_dims(machine2, axis=-1)
    #aug=augment_images(machine1)
    #machine1augmented=np.concatenate([machine1,aug])
    #machine2augmented=np.concatenate([machine2,machine2])
    
    
    totalpatients=np.concatenate([machine1,machine2])
    #labelszero=np.zeros([323]) #When we load only the training set
    #labelsone=np.ones([323])
    labelszero=np.zeros([269]) #When we load only the training set
    labelsone=np.ones([269])
    totallabels=np.concatenate([labelszero,labelsone])
    return(asarray(totalpatients),asarray(totallabels))


###############################################
#Title: Image augmentation for machine learning experiments.
#Author: Aleju
#Date: 2020
#Availability: https://github.com/aleju/imgaug#example_images

def augment_images(a):
    seq = iaa.Sequential([
    iaa.Crop(px=(0, 5)), # crop images from each side by 0 to 16px (randomly chosen)
    #iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
    iaa.Affine(scale={"x": (0.8, 1.1), "y": (0.8, 1.1)},
    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    rotate=(-2, 2),
    shear=(-1, 1))])


    images_aug=[]
    for i in range(a.shape[0]):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
        images = a[i,:,:,0]
        if i==0:
            images_aug= seq(images=images)
        #print('holi')
        #print(images_aug.shape)
            images_aug=expand_dims(images_aug, axis=-1)
        else:
            images_aug=np.dstack((images_aug, seq(images=images)))
        #print('caracoli')
        #print(images_aug.shape)


#images_aug=np.moveaxis(images_aug,[0,1,2],[-1,-2,-3])
    images_aug=np.moveaxis(images_aug,-1,0)
    #print(images_aug.shape)
    images_aug=expand_dims(images_aug, axis=-1)
    #print(images_aug.shape)
    
    return images_aug
###########################################################



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

img_shape=256 #side of the square image
#img_shape=128 #when we crop the image
channels=1 #if it is RGB or B&W
n_classes=2
outputim_shape=128 #side of the square image that the gen outputs



#FOR CASE WHEN WE CROP TO ONLY CENTER
def crop_center(img,cropx,cropy):
    n,y,x,i = img.shape
    #startx = x//4-(cropx//4)
    #starty = y//4-(cropy//4)
    startx=63
    starty=63
    return img[:,starty:starty+cropy,startx:startx+cropx,:]



# load MRI brain images
def load_real_samples():
# load dataset
    (images, labelshort) = load_data1()
    print(images.shape)
    #print(images.shape)
# convert from ints to floats
    #X = expand_dims(images, axis=-1)
    #print(X.shape)
    #X = images.astype('float32')
# scale from [0,255] to [-1,1]
    X = images /np.amax(images)
    #print(labelshort)
    #X = crop_center(X,128,128)
    #plt.imshow(X[10,:,:,0],cmap='gray')
    #print('This is the shape of the dataset ')
    #print(X.shape)
    return [X, labelshort]
 
# # select real samples
def generate_real_samples(dataset):
# split into images and labels
    images, labels = dataset
# choose random instances
    print(images.shape,labels.shape)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=42)
    print(X_train.shape,X_test.shape)
    plt.figure(figsize=(10,10))
    #X_train=augment_images(X_train)

    #print(X_train.shape)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i,:,:,0],cmap='gray')
        plt.xlabel(y_train[i])
    plt.show()
    
    return (X_train, X_test, y_train,y_test)




dataset=load_real_samples()
A,B,Alab,Blab=generate_real_samples(dataset)




model = keras.Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_shape,img_shape,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, padding='same', activation='relu'))#,name='visualized_layer'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, 3, padding='same', activation='relu'))#,name='visualized_layer'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax',name='visualized_layer'))




epochs=1

y_train = to_categorical(Alab, n_classes)
y_test = to_categorical(Blab, n_classes)
datagen = ImageDataGenerator(
    #rotation_range=10,
    #width_shift_range=0.5,
    #height_shift_range=0.5,
    #brightness_range=[0,255],
    #zoom_range=0.5,
    horizontal_flip=True,
    #vertical_flip=True,
)

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
#datagen.fit(A)
# fits the model on batches with real-time data augmentation:
#model.fit(datagen.flow(A, Alab,batch_size=600), epochs=epochs)
model.fit(datagen.flow(A, Alab, batch_size=20),
          steps_per_epoch=20, epochs=epochs)
test_loss, test_acc = model.evaluate(B,  Blab, verbose=2)

print('\nTest accuracy:', test_acc)
predictions = model.predict(B)



model_json = model.to_json()
with open("modeldiscriminatorT1T2.json", "w") as json_file:
    json_file.write(model_json)
    
filename2 = 'modelweightsT1T2.h5'
model.save(filename2)




#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(512, 512)),
#    keras.layers.Dense(32, activation='relu'),
#    keras.layers.Dense(64, activation='relu'),
#    keras.layers.Dense(2, activation='softmax')
#])
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#model.fit(A, Alab, epochs=20)
#test_loss, test_acc = model.evaluate(B,  Blab, verbose=2)

#print('\nTest accuracy:', test_acc)
#predictions = model.predict(B)




def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
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
    #print(true_label)
    thisplot[int(true_label)].set_color('blue')
#################################################################




###############################################
#Title: How to Visualize Filters and Feature Maps in Convolutional Neural Networks
#Author: Jason Brownlee
#Date: 2019
#Availability: https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/




#num_rows = 5
#num_cols = 3
#num_images = num_rows*num_cols
#plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#for i in range(num_images):
#    print(predictions.shape)
#    plt.subplot(num_rows, 2*num_cols, 2*i+1)
#    plot_image(i, predictions[i], Blab, B[:,:,0])
#    plt.subplot(num_rows, 2*num_cols, 2*i+2)
#    plot_value_array(i, predictions[i], Blab)
#plt.tight_layout()
#plt.show()

#print(predictions.shape,predictions)
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    #print(i,predictions[i].shape,Blab.shape,B.shape)
    plot_image(i, predictions[i], Blab, B[i,:,:,0])
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    #print(i,predictions[i,:].shape,Blab.shape)
    plot_value_array(i, predictions[i], Blab)
plt.tight_layout()
plt.show()




# summarize filter shapes
for layer in model.layers:
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
# get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)





# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn of axis
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
# show the figure
plt.show()




# convert the image to an array
ixs = [9]
outputs = [model.layers[i].output for i in ixs]
model1 = Model(inputs=model.inputs, outputs=outputs)
n=1
img = img_to_array(B[n])
plt.imshow(B[n][:,:,0])
print(Blab[n])
#print(img.shape)
#model.summary()
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
#print(img.shape)
# prepare the image (e.g. scale pixel values for the vgg)
#img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model1.predict(img)
#print(feature_maps.shape)
# plot all 64 maps in an 8x8 squares
square = 4
ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(feature_maps[0,:,:,ix-1], cmap='gray')
        ix += 1
        
# show the figure
plt.show()
###############################################################



###############################################
#Title: Visualizing Keras CNN attention: Grad-CAM Class Activation Maps
#Author: Chris
#Date: 2019
#Availability: https://www.machinecurve.com/index.php/2019/11/28/visualizing-keras-cnn-attention-grad-cam-class-activation-maps/




# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'visualized_layer')
# Swap softmax with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model) 
# Numbers to visualize
indices_to_visualize = list(range(0, 22))




# Visualize
for index_to_visualize in indices_to_visualize:
    # Get input
    input_image = B[index_to_visualize]
    input_class = int(Blab[index_to_visualize])
    # Matplotlib preparations
    fig, axes = plt.subplots(1, 3)
    # Generate visualization
    visualization = visualize_cam(model, layer_index, filter_indices=input_class, seed_input=input_image)
    axes[0].imshow(input_image[..., 0], cmap='gray') 
    axes[0].set_title('Input')
    axes[1].imshow(visualization)
    axes[1].set_title('Grad-CAM')
    heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
    original = np.uint8(cm.gray(input_image[..., 0])[..., :3] * 255)
    axes[2].imshow(overlay(heatmap, original))
    axes[2].set_title('Overlay')
    fig.suptitle(f'Brain of scanner = {input_class}')
    #cbar = plt.colorbar(heatmap)
    plt.show()


##############################################################



