
#!/usr/bin/env python
# coding: utf-8

# In[14]:


# example of pix2pix gan for satellite to map image-to-image translation

from matplotlib import pyplot
from keras.preprocessing.image import load_img
import os
import array
import pydicom
import numpy as np
from numpy import expand_dims
from numpy import asarray
import random
# example of training a cyclegan on the horse2zebra dataset
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
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
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import model_from_json
import imgaug
import imgaug.augmenters as iaa
from keras.layers import UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Add
import tensorflow as tf


# In[15]:
def augment_images(a,b):
    seq = iaa.Sequential([
    iaa.Crop(px=(0, 5)), # crop images from each side by 0 to 16px (randomly chosen)
    #iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
    iaa.Affine(scale={"x": (0.8, 1.1), "y": (0.8, 1.1)},
    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    rotate=(-2, 2),
    shear=(-1, 1))])


    images_aug=[]
    image_aug2=[]
    for i in range(a.shape[0]):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
        images = a[i,:,:,0]
        images1 = b[i,:,:,0]
        if i==0:
            images_aug= seq(images=images)
            images_aug2= seq(images=images1)
        #print('holi')
        #print(images_aug.shape)
            images_aug=expand_dims(images_aug, axis=-1)
            images_aug2=expand_dims(images_aug2, axis=-1)
        else:
            images_aug=np.dstack((images_aug, seq(images=images)))
            images_aug2=np.dstack((images_aug2, seq(images=images1)))
        #print('caracoli')
        #print(images_aug.shape)


#images_aug=np.moveaxis(images_aug,[0,1,2],[-1,-2,-3])
    images_aug=np.moveaxis(images_aug,-1,0)
    images_aug2=np.moveaxis(images_aug2,-1,0)
    #print(images_aug.shape)
    images_aug=expand_dims(images_aug, axis=-1)
    images_aug2=expand_dims(images_aug2, axis=-1)
    #print(images_aug.shape)
    
    return images_aug,images_aug2



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
####################################################

def crop_center(img,cropx,cropy):
    n,y,x,i = img.shape
    #startx = x//4-(cropx//4)
    #starty = y//4-(cropy//4)
    startx=63
    starty=63
    return img[:,starty:starty+cropy,startx:startx+cropx,:]

def load_data():
    INPUT_FOLDER='/media/data/penriquez/Images_256'
    #INPUT_FOLDER='/Users/arturo/Documents/MATLAB/THESIS/Images_256'
    #INPUT_FOLDER='/Users/arturo/Desktop/Images_256'
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
                    print('Optima 1 is in the first folder')
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
    aug1,aug2=augment_images(machine1,machine2)
    #print(machine1)
    #print(machine2)
    machine1augmented=np.concatenate([machine1,aug1])
    machine2augmented=np.concatenate([machine2,aug2])
    return(asarray(machine1augmented),asarray(machine2augmented))



###############################################
#Title: How to Develop a CycleGAN for Image-to-Image Translation with Keras
#Author: Jason Brownlee
#Date: 2019
#Availability: https://machinelearningmastery.com/cyclegan-tutorial-with-keras/

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    #d = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    #d = InstanceNormalization(axis=-1)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    # define model
    model = Model(in_image, patch_out)
    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    #g = Conv2D(n_filters, (3,3), strides=(2,2), padding='same')(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
###################################################


###############################################
#Title: How to Develop a Pix2Pix GAN for Image-to-Image Translation
#Author: Jason Brownlee
#Date: 2019
#Availability: https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    #g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(layer_in)

    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model
def define_generator(image_shape=(256,256,1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 32, batchnorm=False)
    e2 = define_encoder_block(e1, 64)
    e3 = define_encoder_block(e2, 128)
    e4 = define_encoder_block(e3, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    #e6 = define_encoder_block(e5, 512)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(1024, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(e5)
    #b = Conv2D(512, (3,3), strides=(2,2), padding='same')(e4)

    b = Conv2D(1024, (3,3), strides=(2,2), padding='same')(e5)
    b = Activation('relu')(b)
    # decoder model
    #d1 = decoder_block(b, e7, 512)
    #d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(b, e5, 512)
    d4 = decoder_block(d3, e4, 256, dropout=False)
    d5 = decoder_block(d4, e3, 128, dropout=False)
    d6 = decoder_block(d5, e2, 64, dropout=False)
    d7 = decoder_block(d6, e1, 32, dropout=False)
    # output
    g = Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    #g = Conv2DTranspose(1, (3,3), strides=(2,2), padding='same')(d7)

    #g = Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

###########################################################

# In[18]:
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    # identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    # forward cycle
    output_f = g_model_2(gen1_out)
    # backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    # define model graph
    def ssim_loss(A,B):
        SSIMloss=((1-tf.image.ssim(A,B,max_val=1))/2)
        return SSIMloss
    model = Model([input_gen, input_id], [output_d,output_f,output_b, output_f, output_b])
    # define optimization algorithm configuration
    opt = Adam(lr=0.0002, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', ssim_loss, ssim_loss, 'mae', 'mae'], loss_weights=[1, 5,5, 10, 10], optimizer=opt)
    return model



# In[21]:

###############################################
#Title: How to Develop a CycleGAN for Image-to-Image Translation with Keras
#Author: Jason Brownlee
#Date: 2019
#Availability: https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
    
# load and prepare training images
def load_real_samples():
    # load the dataset
    data = load_data()
    # unpack arrays
    X1, X2 = data[0], data[1]
    # scale from [0,255] to [-1,1]
    X1 = X1/np.amax(X1)
    X2 = X2/np.amax(X2)
    X1 = crop_center(X1,128,128)
    X2=crop_center(X2,128,128)
    print(np.min(X1),np.amax(X1),np.min(X2),np.amax(X2))
    #X1=np.where(X1<=0, 0, X1)
    #X2=np.where(X2<=0, 0, X2)
    return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset1,dataset2, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, dataset1.shape[0], n_samples)
    # retrieve selected images
    X = dataset1[ix]
    Y=dataset2[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, Y, y, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    #pyplot.figure()
    #pyplot.imshow(X[0,:,:,0],cmap='gray')
    #pyplot.show()
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def save_models(step, g_model_AtoB, g_model_BtoA):
    # save the first generator model
    filename1 = 'results/g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
# save the second generator model
    filename2 = 'results/g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# In[22]:


def summarize_performance(step, g_model1, g_model2, trainX, trainXB, name, nameB, n_samples=5):
    # select a sample of input images
    X_in, X_inB,_,_ = generate_real_samples(trainX,trainXB, n_samples, 0)
    # generate translated images
    X_out, _ = generate_fake_samples(g_model1, X_in, 0)
    X_outB, _ = generate_fake_samples(g_model2, X_inB, 0)
    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in+1) /2.0   #Esto estaba diferente en el primer run asi que se guardan mal los resultados
    X_inB = (X_inB+1) /2.0
    X_out = (X_out+1)/2.0
    X_outB = (X_outB+1)/2.0
    # plot real images
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_in[i,:,:,0])
    # plot translated image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_out[i,:,:,0])
    filename1 = 'results/%s_generated_plot_%06d.png' % (name, (step+1))
    pyplot.savefig(filename1)
    pyplot.close()
    
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_inB[i,:,:,0])
    # plot translated image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_outB[i,:,:,0])
    # save plot to file
    
    filename2 = 'results/%s_generated_plot_%06d.png' % (nameB, (step+1))
    pyplot.savefig(filename2)
    pyplot.close()

# In[23]:


# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

# create a line plot of loss for the gan and save to file
def plot_history(dA1_hist, dA2_hist, dB1_hist,dB2_hist,g1_hist,g2_hist):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(dA1_hist, label='d-real1')
    pyplot.plot(dB1_hist, label='d-real2')
    pyplot.plot(dA2_hist, label='d-fake1')
    pyplot.plot(dB2_hist, label='d-fake1')
    pyplot.plot(g1_hist, label='gen1')
    pyplot.plot(g2_hist, label='gen1')
    pyplot.legend()
    # plot discriminator accuracy
    #pyplot.subplot(2, 1, 2)
    #pyplot.plot(a1_hist, label='acc-real')
    #pyplot.plot(a2_hist, label='acc-fake')
    #pyplot.legend()
    # save plot to file
    filename3 = 'results/plot_line_plot_loss.png'
    pyplot.savefig(filename3)
    pyplot.close()

# In[24]:


def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    # define properties of the training run
    n_epochs, n_batch, = 20, 5
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    dA1_hist, dA2_hist, dB1_hist, dB2_hist, g1_hist, g2_hist = list(), list(), list(), list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, X_realB, y_realA,y_realB = generate_real_samples(trainA,trainB, n_batch, n_patch)
# generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        # update generator B->A via adversarial and cycle loss
        g_loss2,disclossBA, SSIMloss1BA, SSIMloss2BA, cycleloss1BA ,cycleloss2BA = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA,X_realB, X_realA , X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1,disclossAB, SSIMloss1AB, SSIMloss2AB,  cycleloss1AB ,cycleloss2AB= c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realA, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        # summarize performance
        print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f] disclossAB[%.3f,%.3f]  SSIM1AB[%.3f,%.3f] SSIM1BA[%.3f,%.3f]  cycleloss1BA[%.3f,%.3f] cycleloss1AB[%.3f,%.3f] ' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2,disclossAB,disclossBA, SSIMloss1AB,SSIMloss2AB, SSIMloss1BA,SSIMloss2BA, cycleloss1BA, cycleloss2BA, cycleloss1AB, cycleloss2AB))
        
        dA1_hist.append(dA_loss1)
        dA2_hist.append(dA_loss2)
        dB1_hist.append(dB_loss1)
        dB2_hist.append(dB_loss2)
        g1_hist.append(g_loss1)
        g2_hist.append(g_loss2)
        # evaluate the model performance every so often
        plot_history(dA1_hist, dA2_hist, dB1_hist,dB2_hist,g1_hist,g2_hist)
        if (i+1) % (30) == 0:
            # plot A->B translation
            save_models(i, g_model_AtoB, g_model_BtoA)
            summarize_performance(i, g_model_AtoB,g_model_BtoA,trainA, trainB,'AtoB','BtoA')
        #if (i+1) % (bat_per_epo * 5) == 0:
            # save the models
        #    save_models(i, g_model_AtoB, g_model_BtoA)
        #    summarize_performance(i, g_model_AtoB,g_model_BtoA,trainA, trainB,'AtoB','BtoA')
        if i == n_steps-10:
            save_models(i, g_model_AtoB, g_model_BtoA)
            summarize_performance(i, g_model_AtoB,g_model_BtoA,trainA, trainB,'AtoB','BtoA')
        if g_loss1<=1.0 and g_loss1<=1.0:
            save_models(i, g_model_AtoB, g_model_BtoA)
            summarize_performance(i, g_model_AtoB,g_model_BtoA,trainA, trainB,'AtoB','BtoA')
        if dA_loss2>=0.3 or dB_loss2>=0.3:
            save_models(i, g_model_AtoB, g_model_BtoA)
            summarize_performance(i, g_model_AtoB,g_model_BtoA,trainA, trainB,'AtoB','BtoA')



# In[25]:


# load image data
dataset = load_real_samples()
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
###################################################



