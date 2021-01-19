#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk

import numpy as np
import os
OUTPUT_DIR = 'output'
import scipy.misc
from PIL import Image
#import gui

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.preprocessing.image import load_img
import array
import pydicom 
from numpy import expand_dims
from numpy import asarray
import tensorflow as tf
import sklearn
from sklearn.metrics import mean_absolute_error
from ipywidgets import interact, fixed
import itk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
PixelType = itk.ctype("double")

from skimage import io
from image_registration import register_images
from scipy.ndimage import shift


# In[2]:


path='/Users/arturo/Desktop/Images_256'
pathpatients=[f for f in os.listdir(path) if not f.startswith('.')]
n=10
for i in pathpatients:
    pathmachines=os.path.join(path, i)
    machines=[f for f in os.listdir(pathmachines) if not f.startswith('.')]
    
    pathsequences1=os.path.join(pathmachines, machines[0])
    sequences1=[m for m in os.listdir(pathsequences1) if not m.startswith('.')]
    sequences1.sort
    sequences1=[a for a in sequences1 if str.find(a,"T1Mapping")>=0]
    
    pathsequences2=os.path.join(pathmachines, machines[1])
    sequences2=[m for m in os.listdir(pathsequences2) if not m.startswith('.')]
    sequences2.sort
    sequences2=[a for a in sequences2 if str.find(a,"T1Mapping")>=0]
    #print(pathsequences1)
    pathimages1=os.path.join(pathsequences1, sequences1[0])
    pathimages2=os.path.join(pathsequences2, sequences2[0])
    #print(pathimages1)
    images1=[f for f in os.listdir(pathimages1) if not f.startswith('.') ]
    images2=[f for f in os.listdir(pathimages2) if not f.startswith('.') ]
    images1=sorted([x for x in images1])
    images2=sorted([x for x in images2])
    #print(images1)
    #raise
    if os.path.exists('%s/regist'%(pathimages1)):
        images1.remove('regist')
    if os.path.exists('%s/regist'%(pathimages2)):
        images2.remove('regist')
    if 'v_headers'in images1:
        images1.remove('v_headers')
        images1.remove('v_headers.index')
    if 'v_headers'in images2:
        images2.remove('v_headers')
        images2.remove('v_headers.index')
    #print(images1)
    #print(images2)
    #raise
    #slices1 = [pydicom.dcmread(pathimages1 + "/" + s,force="True") for s in os.listdir(pathimages1)]
    #slices2 = [pydicom.dcmread(pathimages2 + "/" + s,force="True") for s in os.listdir(pathimages2)]
    
    #print(images1)
    #print(images2)
    #print(images1)
    for j,k in zip(images1,images2):
        #print()
        path1=os.path.join(pathimages1, j)
        path2=os.path.join(pathimages2, k)
        #print(path1)
        #print(path1)
        
        a=pydicom.dcmread(path1,force=True)
        b=pydicom.dcmread(path2,force=True)
        
        machine1=a.ManufacturerModelName
        machine2=b.ManufacturerModelName
        
        if not os.path.exists('%s/regist'%(pathimages1)):
            os.makedirs('%s/regist'%(pathimages1))
        if not os.path.exists('%s/regist'%(pathimages2)):
            os.makedirs('%s/regist'%(pathimages2))
        
        topath1='%s/regist/%s%d.dcm'%(pathimages1,machine1,n)
        topath2='%s/regist/%s%d.dcm'%(pathimages2,machine2,n)
        
        #raise
        
        #print(path1,path2)
        
        a1=a.pixel_array
        b1=b.pixel_array
        
        #a1 = (255.0 / a1.max() * (a1 - a1.min())).astype(np.uint16)
        #b1 = (255.0 / b1.max() * (b1 - b1.min())).astype(np.uint16)
        
        #print(np.amax(a),np.amax(b))

        posx,posy=register_images(a1,b1)
        a1new=shift(a1,shift=(-posx/2,-posy/2),mode='constant')
        b1new=shift(b1,shift=(posx/2,posy/2),mode='constant')
        raise
        #print(np.min(a1),np.min(correctedimage))
        a1new=np.where(a1new<=0, 0, a1new)
        b1new=np.where(b1new<=0, 0, b1new)
        #raise
        #print(posx,posy)
        a.PixelData=a1new
        a.save_as(topath1)
        b.PixelData=b1new
        b.save_as(topath2)
        
        
        n=n+1
        #raise
        #raise
        
            
        
   
