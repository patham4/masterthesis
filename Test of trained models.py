#!/usr/bin/env python
# coding: utf-8



import h5py
# example of pix2pix gan for satellite to map image-to-image translation
import tensorflow as tf
import tensorflow.keras as keras
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
import os
import array
import pydicom 
import numpy as np
from numpy import expand_dims
from numpy import asarray
from keras.models import model_from_json

# Random Shifts
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
from skimage.metrics import structural_similarity as ssim
import cv2

from skimage.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse
import sklearn.feature_selection
from sklearn.preprocessing import normalize




#Load registered images



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
##########################################################



def load_data():
    #INPUT_FOLDER='/media/data/penriquez/Images_256'
    #INPUT_FOLDER='/Users/arturo/Desktop/Images_256'
    INPUT_FOLDER='/Users/arturo/Desktop/Images_test 2'
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
    return(asarray(machine1),asarray(machine2))




#Load unregistered images



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


#############################################################


def load_data1():
    #INPUT_FOLDER='/media/data/penriquez/Images_256_unregistered'
    #INPUT_FOLDER='/Users/arturo/Documents/MATLAB/THESIS/Images_256_unregistered'
    INPUT_FOLDER='/Users/arturo/Documents/MATLAB/THESIS/Images_test'
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
            #sequences3=[a for a in sequences if str.find(a,"T2MapPreset-1")>=0]
            #REST OF RUNS
            sequences3=[a for a in sequences if str.find(a,"Mapping")>=0]
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
    #aug=augment_images(machine1)
    #machine1augmented=np.concatenate([machine1,aug])
    #machine2augmented=np.concatenate([machine2,machine2])
    return(asarray(machine1),asarray(machine2))




def crop_center(img,cropx,cropy):
    n,y,x,i = img.shape
    #startx = x//4-(cropx//4)
    #starty = y//4-(cropy//4)
    startx=63
    starty=63
    return img[:,starty:starty+cropy,startx:startx+cropx,:]




#UNREGISTERED RUNS



#Exp1
json_fileexp1AB = open('exp1/modelg1.json', 'r')
json_fileexp1BA= open('exp1/modelg2.json', 'r')
#Exp2
json_fileexp2AB = open('exp2/modelg1.json', 'r')
json_fileexp2BA= open('exp2/modelg2.json', 'r')
#Exp3
json_fileexp3AB = open('exp3/modelg1.json', 'r')
json_fileexp3BA= open('exp3/modelg2.json', 'r')
#Exp4
json_fileexp4AB = open('exp4/modelg1.json', 'r')
json_fileexp4BA= open('exp4/modelg2.json', 'r')
#Exp5
json_fileexp6AB = open('exp6/modelg1.json', 'r')
json_fileexp6BA= open('exp6/modelg2.json', 'r')


#Exp1
loaded_model_jsonexp1AB = json_fileexp1AB.read()
loaded_model_jsonexp1BA = json_fileexp1BA.read()
#Exp2
loaded_model_jsonexp2AB = json_fileexp2AB.read()
loaded_model_jsonexp2BA = json_fileexp2BA.read()
#Exp3
loaded_model_jsonexp3AB = json_fileexp3AB.read()
loaded_model_jsonexp3BA = json_fileexp3BA.read()
#Exp4
loaded_model_jsonexp4AB = json_fileexp4AB.read()
loaded_model_jsonexp4BA = json_fileexp4BA.read()
#Exp6
loaded_model_jsonexp6AB = json_fileexp6AB.read()
loaded_model_jsonexp6BA = json_fileexp6BA.read()


#Exp1
json_fileexp1AB.close()
json_fileexp1BA.close()
#Exp2
json_fileexp2AB.close()
json_fileexp2BA.close()
#Exp3
json_fileexp3AB.close()
json_fileexp3BA.close()
#Exp4
json_fileexp4AB.close()
json_fileexp4BA.close()
#Exp6
json_fileexp6AB.close()
json_fileexp6BA.close()


#Exp1
loaded_modelexp1AB = model_from_json(loaded_model_jsonexp1AB)
loaded_modelexp1BA = model_from_json(loaded_model_jsonexp1BA)
#Exp2
loaded_modelexp2AB = model_from_json(loaded_model_jsonexp2AB)
loaded_modelexp2BA = model_from_json(loaded_model_jsonexp2BA)
#Exp3
loaded_modelexp3AB = model_from_json(loaded_model_jsonexp3AB)
loaded_modelexp3BA = model_from_json(loaded_model_jsonexp3BA)
#Exp4
loaded_modelexp4AB = model_from_json(loaded_model_jsonexp4AB)
loaded_modelexp4BA = model_from_json(loaded_model_jsonexp4BA)
#Exp6
loaded_modelexp6AB = model_from_json(loaded_model_jsonexp6AB)
loaded_modelexp6BA = model_from_json(loaded_model_jsonexp6BA)



#Exp1
loaded_modelexp1AB.load_weights("exp1/g_model_AtoB_000720.h5")
loaded_modelexp1BA.load_weights("exp1/g_model_BtoA_000720.h5")
#Exp2
loaded_modelexp2AB.load_weights("exp2/g_model_AtoB_000720.h5")
loaded_modelexp2BA.load_weights("exp2/g_model_BtoA_000720.h5")
#Exp3
loaded_modelexp3AB.load_weights("exp3/g_model_AtoB_003230.h5")
loaded_modelexp3BA.load_weights("exp3/g_model_BtoA_003230.h5")
#Exp4
loaded_modelexp4AB.load_weights("exp4/g_model_AtoB_001290.h5")
loaded_modelexp4BA.load_weights("exp4/g_model_BtoA_001290.h5")
#Exp6
loaded_modelexp6AB.load_weights("exp6/g_model_AtoB_001370.h5")
loaded_modelexp6BA.load_weights("exp6/g_model_BtoA_001370.h5")



print("Loaded model from disk")
data = load_data1()
# unpack arrays
X1, X2 = data[0], data[1]
print(X1.shape,X2.shape)
# scale from [0,255] to [-1,1]
X1 = X1 /np.amax(X1)
X2 = X2 /np.amax(X2)
#X1 = crop_center(X1,128,128)
#X2=crop_center(X2,128,128)
##X1=normalize(X1)
#X2=normalize(X2)
X1=np.where(X1<=0, 0, X1)
X2=np.where(X2<=0, 0, X2)

opt = Adam(lr=0.0002, beta_1=0.5)
#Exp1
loaded_modelexp1AB.compile(loss='binary_crossentropy', optimizer=opt)
loaded_modelexp1BA.compile(loss='binary_crossentropy', optimizer=opt)
#Exp2
loaded_modelexp2AB.compile(loss='binary_crossentropy', optimizer=opt)
loaded_modelexp2BA.compile(loss='binary_crossentropy', optimizer=opt)
#Exp3
loaded_modelexp3AB.compile(loss='binary_crossentropy', optimizer=opt)
loaded_modelexp3BA.compile(loss='binary_crossentropy', optimizer=opt)
#Exp4
loaded_modelexp4AB.compile(loss='binary_crossentropy', optimizer=opt)
loaded_modelexp4BA.compile(loss='binary_crossentropy', optimizer=opt)
#Exp6
loaded_modelexp6AB.compile(loss='binary_crossentropy', optimizer=opt)
loaded_modelexp6BA.compile(loss='binary_crossentropy', optimizer=opt)


#Exp1
scorerunexp1AB = loaded_modelexp1AB.predict(X1)
scorerunexp1BA=loaded_modelexp1BA.predict(X2)
#Exp2
scorerunexp2AB = loaded_modelexp2AB.predict(X1)
scorerunexp2BA= loaded_modelexp2BA.predict(X2)
#Exp3
scorerunexp3AB = loaded_modelexp3AB.predict(X1)
scorerunexp3BA=loaded_modelexp3BA.predict(X2)
#Exp4
scorerunexp4AB = loaded_modelexp4AB.predict(X1)
scorerunexp4BA=loaded_modelexp4BA.predict(X2)
#Exp6
scorerunexp6AB = loaded_modelexp6AB.predict(X1)
scorerunexp6BA=loaded_modelexp6BA.predict(X2)


scorerunexp1AB=np.where(scorerunexp1AB<=0, 0, scorerunexp1AB)
scorerunexp1BA=np.where(scorerunexp1BA<=0, 0, scorerunexp1BA)

scorerunexp2AB=np.where(scorerunexp2AB<=0, 0, scorerunexp2AB)
scorerunexp2BA=np.where(scorerunexp2BA<=0, 0, scorerunexp2BA)

scorerunexp3AB=np.where(scorerunexp3AB<=0, 0, scorerunexp3AB)
scorerunexp3BA=np.where(scorerunexp3BA<=0, 0, scorerunexp3BA)

scorerunexp4AB=np.where(scorerunexp4AB<=0, 0, scorerunexp4AB)
scorerunexp4BA=np.where(scorerunexp4BA<=0, 0, scorerunexp4BA)

scorerunexp6AB=np.where(scorerunexp6AB<=0, 0, scorerunexp6AB)
scorerunexp6BA=np.where(scorerunexp6BA<=0, 0, scorerunexp6BA)




#REGISTERED RUN
#Exp1
json_fileexp5AB = open('exp5/modelg1.json', 'r')
json_fileexp5BA= open('exp5/modelg2.json', 'r')

loaded_model_jsonexp5AB = json_fileexp5AB.read()
loaded_model_jsonexp5BA = json_fileexp5BA.read()

json_fileexp5AB.close()
json_fileexp5BA.close()

loaded_modelexp5AB = model_from_json(loaded_model_jsonexp5AB)
loaded_modelexp5BA = model_from_json(loaded_model_jsonexp5BA)

loaded_modelexp5AB.load_weights("exp5/g_model_AtoB_001474.h5")
loaded_modelexp5BA.load_weights("exp5/g_model_BtoA_001474.h5")


print("Loaded model from disk")
data = load_data()
# unpack arrays
X1, X2 = data[0], data[1]
print(X1.shape,X2.shape)
# scale from [0,255] to [-1,1]
X1 = X1 /np.amax(X1)
X2 = X2 /np.amax(X2)
#X1 = crop_center(X1,128,128)
#X2=crop_center(X2,128,128)
##X1=normalize(X1)
#X2=normalize(X2)
X1=np.where(X1<=0, 0, X1)
X2=np.where(X2<=0, 0, X2)

opt = Adam(lr=0.0002, beta_1=0.5)
#Exp1
loaded_modelexp5AB.compile(loss='binary_crossentropy', optimizer=opt)
loaded_modelexp5BA.compile(loss='binary_crossentropy', optimizer=opt)

scorerunexp5AB = loaded_modelexp5AB.predict(X1)
scorerunexp5BA=loaded_modelexp5BA.predict(X2)

scorerunexp5AB=np.where(scorerunexp5AB<=0, 0, scorerunexp5AB)
scorerunexp5BA=np.where(scorerunexp5BA<=0, 0, scorerunexp5BA)





#RUN FOR INITIAL VALIDATION OF DATASETS
data = load_data()
# unpack arrays
X1, X2 = data[0], data[1]
print(X1.shape,X2.shape)
X1 = X1 /np.amax(X1)
X2 = X2 /np.amax(X2)
X1=np.where(X1<=0, 0, X1)
X2=np.where(X2<=0, 0, X2)
print(np.max(X1),np.min(X1),np.max(X2),np.min(X2))




print(np.max(scorerunexp6AB),np.min(scorerunexp6AB),np.max(scorerunexp6BA),np.min(scorerunexp6BA))
print(np.max(X1),np.min(X1),np.max(X2),np.min(X2))




#VALIDATION OF DATASETS
sum_num=0
sum_nummse=0
for i in range(X2.shape[0]):
    sum_num=sum_num+ssim(X1[i,:,:,0],X2[i,:,:,0])
    sum_nummse=sum_nummse+normalized_root_mse(X1[i,:,:,0],X2[i,:,:,0])
    print('%.3f %.3f'% (ssim(X1[i,:,:,0],X2[i,:,:,0]),normalized_root_mse(X1[i,:,:,0],X2[i,:,:,0])))
avg = sum_num / (X2.shape[0])
avgmse=sum_nummse/X2.shape[0]
print('Average %.3f %.3f' % (avg,avgmse))



#Average histogram of Scanner A
a=X1[0,:,:,0]
for i in range(X1.shape[0]):
    #print(i)
    #print(a,X1[i,:,:,0])
    #print(i)
    a = cv2.add(a,X1[i,:,:,0])
#print(a.shape)
a=a/X1.shape[0]
b=np.hstack(a)
hist1 = pyplot.hist(b, bins=50)  # arguments are passed to np.histogram
#Text(0.5, 1.0, "Histogram with 'auto' bins")
pyplot.show()




#Average histogram of scanner B
a=X2[0,:,:,0]
for i in range(X1.shape[0]):
    #print(a,X1[i,:,:,0])
    a = cv2.add(a,X2[i,:,:,0])
#print(a.shape)
a=a/X1.shape[0]
c=np.hstack(a)
hist2= pyplot.hist(c, bins=50)  # arguments are passed to np.histogram
#Text(0.5, 1.0, "Histogram with 'auto' bins")
pyplot.show()


# In[577]:


bins=50
#Side by side histograms of scanners A and B
pyplot.hist([b, c], bins, label=['A', 'B'],range=[0.1,0.5])
pyplot.legend(loc='upper right')
pyplot.show()


# In[621]:


#histogram of fake A images
a=scorerunexp4AB[0,:,:,0]
for i in range(scorerunexp4AB.shape[0]):
    #print(a,X1[i,:,:,0])
    a = cv2.add(a,scorerunexp4AB[i,:,:,0])
#print(a.shape)
a=a/scorerunexp4AB.shape[0]
d=np.hstack(a)
_ = pyplot.hist(d, bins=50)  # arguments are passed to np.histogram
#Text(0.5, 1.0, "Histogram with 'auto' bins")
pyplot.show()


# In[622]:


#hisogram of fake B images
a=scorerunexp4BA[0,:,:,0]
for i in range(scorerunexp4BA.shape[0]):
    #print(a,X1[i,:,:,0])
    a = cv2.add(a,scorerunexp4BA[i,:,:,0])
#print(a.shape)
a=a/scorerunexp4BA.shape[0]
e=np.hstack(a)
_ = pyplot.hist(e, bins=50)  # arguments are passed to np.histogram
#Text(0.5, 1.0, "Histogram with 'auto' bins")
pyplot.show()



# In[623]:


#Side by side histograms of scanners A and fakeA
pyplot.hist([b,c, d], bins, label=['A', 'B','fakeB'],range=[0.1,0.5])
pyplot.legend(loc='upper right')
pyplot.show()


# In[624]:


#Side by side histograms of scanners A and fakeB
pyplot.hist([b, c, e], bins, label=['A', 'B','fakeA'],range=[0.1,0.5])
pyplot.legend(loc='upper right')
pyplot.show()


# In[625]:


#Side by side histograms of scanners A and fakeB
pyplot.hist([e,d], bins, label=['fakeA', 'fakeB'],range=[0.1,0.5])
pyplot.legend(loc='upper right')
pyplot.show()


# In[626]:


pyplot.plot(X1.ravel(), scorerunexp4BA.ravel(), '.')
pyplot.xlabel('A')
pyplot.ylabel('fakeA')
#pyplot.title('T1 vs T2 signal')
np.corrcoef(X1.ravel(), scorerunexp4BA.ravel())[0, 1]


# In[627]:


pyplot.plot(X1.ravel(), scorerunexp4AB.ravel(), '.')
pyplot.xlabel('A')
pyplot.ylabel('fakeB')
#pyplot.title('T1 vs T2 signal')
np.corrcoef(X1.ravel(), scorerunexp4AB.ravel())[0, 1]


# In[628]:


pyplot.plot(X2.ravel(), scorerunexp4AB.ravel(), '.')
pyplot.xlabel('B')
pyplot.ylabel('fakeB')
#pyplot.title('T1 vs T2 signal')
np.corrcoef(X2.ravel(), scorerunexp4AB.ravel())[0, 1]


# In[629]:


pyplot.plot(X2.ravel(), scorerunexp4BA.ravel(), '.')
pyplot.xlabel('B')
pyplot.ylabel('fakeA')
#pyplot.title('T1 vs T2 signal')
np.corrcoef(X2.ravel(), scorerunexp4BA.ravel())[0, 1]


# In[630]:


pyplot.plot(X1.ravel(), X2.ravel(), '.')
pyplot.xlabel('A')
pyplot.ylabel('B')
#pyplot.title('T1 vs T2 signal')
np.corrcoef(X1.ravel(), X2.ravel())[0, 1]


# In[631]:


pyplot.plot(scorerunexp4BA.ravel(), scorerunexp4AB.ravel(), '.')
pyplot.xlabel('fakeA')
pyplot.ylabel('fakeB')
#pyplot.title('T1 vs T2 signal')
np.corrcoef(scorerunexp4BA.ravel(), scorerunexp4AB.ravel())[0, 1]


#SAVE THE GENERATED IMAGES TO A PICKLE TO PASS THEM LATER TO THE TRAINED CLASSIFIER

with open('exp4/objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(scorerunexp4AB, f)
    f.close()




