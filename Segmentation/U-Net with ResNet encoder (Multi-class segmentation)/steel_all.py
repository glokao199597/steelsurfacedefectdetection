
 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:45:24 2020

@author: Ching
"""
#double brackets return dataframes

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.layers import *
from keras.applications import *
from keras.layers.core import Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from sklearn.metrics import accuracy_score
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import *
from keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report,precision_recall_fscore_support
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, History
from PIL import Image
from PIL import ImageFilter
from skimage.io import imshow
from skimage.util import random_noise
from sklearn.utils import class_weight
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, precision_recall_curve,recall_score,average_precision_score,auc
import segmentation_models as sm
import skimage.transform  
import itertools
import albumentations
from ImageDataAugmentor.image_data_augmentor import *
import json
from cosine_annealing import CosineAnnealingScheduler
from accum_optimizer import AccumOptimizer

keras.backend.set_image_data_format('channels_last')

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import , Dense, BatchNormalization, Dropout
seed = 42
size = 0

def ImageSave(x_tst, true_y, preds_train, aug_type):
    for i in range(3):
        plt.subplot(3, 3, 1 + i)
        plt.axis('off')
        plt.imshow(x_tst[i].astype(np.int32))
        #plt.imshow(x_tst[i],cmap='gray')
    for i in range(3): 
        plt.subplot(3, 3, 1 + 3 + i)
        plt.axis('off')
        plt.title('Class:' + ' '.join(map(str, np.unique(true_y[i]))))
        plt.imshow(true_y[i])
    # plot real target image
    for i in range(3):
        plt.subplot(3, 3, 1 + 3*2 + i)
        plt.axis('off')
        plt.title('Class:' + ' '.join(map(str, np.unique(preds_train[i]))))
        plt.imshow(preds_train[i])
    filename1 = 'ALL_' + aug_type + '1' + '.png'
    plt.savefig(filename1)
    plt.close()
    idx = 0
    for i in range(4,7):
        plt.subplot(3, 3, 1 + idx)
        plt.axis('off')
        plt.imshow(x_tst[i].astype(np.int32))
        #plt.imshow(x_tst[i],cmap='gray')
        idx = idx+1
    # plot generted target image
    idx = 0
    for i in range(4,7):
        plt.subplot(3, 3, 1 + 3 + idx)
        plt.axis('off')
        plt.title('Class:' + ' '.join(map(str, np.unique(true_y[i]))))
        plt.imshow(true_y[i])
        idx = idx+1
    # plot real target image
    idx = 0
    for i in range(4,7):
        plt.subplot(3, 3, 1 + 3*2 + idx)
        plt.axis('off')
        plt.title('Class:' + ' '.join(map(str, np.unique(preds_train[i]))))
        plt.imshow(preds_train[i])
        idx = idx+1
    filename1 = 'ALL_' + aug_type + '2' + '.png'
    plt.savefig(filename1)
    plt.close()
    idx = 0
    for i in range(8,11):
        plt.subplot(3, 3, 1 + idx)
        plt.axis('off')
        plt.imshow(x_tst[i].astype(np.int32))
        #plt.imshow(x_tst[i],cmap='gray')
        idx = idx+1
    # plot generted target image
    idx = 0
    for i in range(8,11):
        plt.subplot(3, 3, 1 + 3 + idx)
        plt.axis('off')
        plt.title('Class:' + ' '.join(map(str, np.unique(true_y[i]))))
        plt.imshow(true_y[i])
        idx = idx+1
    # plot real target image
    idx = 0
    for i in range(8,11):
        plt.subplot(3, 3, 1 + 3*2 + idx)
        plt.axis('off')
        plt.title('Class:' + ' '.join(map(str, np.unique(preds_train[i]))))
        plt.imshow(preds_train[i])
        idx = idx+1
    filename1 = 'ALL_' + aug_type + '3' + '.png'
    plt.savefig(filename1)
    plt.close()
    idx = 0
    counter = 0
    for i in range(9,len(true_y)):
        if(idx==3):
            idx = 0
            plt.savefig('steel_ALL_' + aug_type + '%d.png' % (counter))
            plt.close()
            counter = counter + 1
        if(counter == 3):
            break
        if(len(np.unique(true_y[i]))>2): 
            plt.subplot(3, 3, 1 + idx)
            plt.axis('off')
            plt.imshow(x_tst[i].astype(np.int32))

            plt.subplot(3, 3, 1 + 3 + idx)
            plt.axis('off')
            plt.title('Class:' + ' '.join(map(str, np.unique(true_y[i]))))
            plt.imshow(true_y[i])

            plt.subplot(3, 3, 1 + 3*2 + idx)        
            plt.axis('off')
            plt.title('Class:' + ' '.join(map(str, np.unique(preds_train[i]))))
            plt.imshow(preds_train[i])
            idx = idx+1 


def my_metrics(y_true, y_pred, i):
    y_true_pos = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,i])
    y_pred_pos = K.flatten(y_pred[...,i])
    true_pos = K.sum(y_true_pos * y_pred_pos,axis=-1)
    true_neg = K.sum((1-y_true_pos) * (1-y_pred_pos),axis=-1)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos),axis=-1)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos,axis=-1)
    accuracy = (true_pos+true_neg)/(true_neg+true_pos+false_neg+false_pos)
    recall = true_pos / (true_pos+false_neg)
    precision = true_pos / (true_pos+true_neg)
    f1 = (2 * true_pos) / (2*true_pos+false_pos+false_neg)
    return recall, precision,f1

def sum_enc(i):
    return sum([int(k) for k in i.split(' ')[1::2]])



def mask2rle(img):
    img = np.squeeze(img, axis=0)
    img = np.squeeze(img, axis=2)
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, i, shape=(1600,256)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype =np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = i
    return img.reshape(shape).T

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 1)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)



def to_one_hot(y_true, y_pred):
    y_pred_pos = K.one_hot(K.cast(y_pred, 'int32'), num_classes=5)
    y_true_pos = K.one_hot(K.cast(y_true, 'int32'), num_classes=5)
    return y_true_pos, y_pred_pos



def to_one_hot_avg(y_true,y_pred):
    y_true_pos = K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,1:]
    y_pred_pos = K.one_hot(K.cast(y_pred, 'int32'), num_classes=5)[...,1:]
    return y_true_pos,y_pred_pos


def tversky(y_true, y_pred, smooth=1e-6, alpha=0.7):
    #y_true_pos = K.flatten(y_true)
    #y_pred_pos = K.flatten(y_pred)
    y_true_pos = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,1:])
    y_pred_pos = K.flatten(y_pred[...,1:])
    true_pos = K.sum(y_true_pos * y_pred_pos,axis=-1)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos),axis=-1)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos,axis=-1)
    return (true_pos + smooth) / (true_pos + 0.7 * false_neg + 0.3 * false_pos + smooth)


def tversky1(y_true, y_pred, smooth=1e-6, alpha=0.7):
    #y_true_pos = K.flatten(y_true)
    #y_pred_pos = K.flatten(y_pred)
    y_true_pos = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,1])
    y_pred_pos = K.flatten(y_pred[...,1])
    true_pos = K.sum(y_true_pos * y_pred_pos,axis=-1) 
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos),axis=-1)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos,axis=-1)
    return (true_pos + smooth) / (true_pos + 0.7 * false_neg + 0.3 * false_pos + smooth)

def tversky2(y_true, y_pred, smooth=1e-6, alpha=0.7):
    #y_true_pos = K.flatten(y_true)
    #y_pred_pos = K.flatten(y_pred)
    y_true_pos = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,2])
    y_pred_pos = K.flatten(y_pred[...,2])
    true_pos = K.sum(y_true_pos * y_pred_pos,axis=-1)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos),axis=-1)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos,axis=-1)
    return (true_pos + smooth) / (true_pos + 0.7 * false_neg + 0.3 * false_pos + smooth)

def tversky3(y_true, y_pred, smooth=1e-6, alpha=0.7):
    #y_true_pos = K.flatten(y_true)
    #y_pred_pos = K.flatten(y_pred)
    y_true_pos = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,3])
    y_pred_pos = K.flatten(y_pred[...,3])
    true_pos = K.sum(y_true_pos * y_pred_pos,axis=-1)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos),axis=-1)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos,axis=-1)
    return (true_pos + smooth) / (true_pos + 0.7 * false_neg + 0.3 * false_pos + smooth)

def tversky4(y_true, y_pred, smooth=1e-6, alpha=0.7):
    #y_true_pos = K.flatten(y_true)
    #y_pred_pos = K.flatten(y_pred)
    y_true_pos = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,4])
    y_pred_pos = K.flatten(y_pred[...,4])
    true_pos = K.sum(y_true_pos * y_pred_pos,axis=-1)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos),axis=-1)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos,axis=-1)
    return (true_pos + smooth) / (true_pos + 0.7 * false_neg + 0.3 * false_pos + smooth)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 3) 

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat1(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,1])
    y_pred_f = K.flatten(y_pred[...,1])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat2(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,2])
    y_pred_f = K.flatten(y_pred[...,2])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat3(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,3])
    y_pred_f = K.flatten(y_pred[...,3])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat4(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 1 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=5)[...,4])
    y_pred_f = K.flatten(y_pred[...,4])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_9cat(y_true, y_pred)

def inverse_trasform(train_data):
    new = train_data*(np.array([0.19438064, 0.19438064, 0.19438064]))
    new = new+(np.array([0.36270394, 0.36270394, 0.36270394]))
    #X_train = X_train/255.0
    '''new = train_data*np.array([0.229, 0.224, 0.225])
    new = new + np.array([0.485, 0.456, 0.406])'''
    #X_train = X_train.astype(np.float32)
    return new

def multi_tversky(y_true, y_pred):
    alpha = 0.7
    beta  = 0.3
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def multi_tversky_loss(y_true, y_pred):
    return 1 - multi_tversky(y_true,y_pred)

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
'''def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)'''

train_path = 'severstal-steel-defect-detection/train_images/'
test_path = 'severstal-steel-defect-detection/test_images/'

train_images = next(os.walk(train_path))[2]
train_images.sort()
test_images = next(os.walk(test_path))[2]
cnt = 0


train_images_df = pd.DataFrame(train_images,columns =['ImageId'])
train_mask_df = pd.read_csv("severstal-steel-defect-detection/train.csv")
temp_train_data = train_mask_df.merge(train_images_df, how='outer',on='ImageId')
temp_train_data['ClassId'] = temp_train_data['ClassId'].fillna(0)
#temp_train_data['ClassId'].value_counts().sort_index(ascending=True).plot('bar')
#plt.show()
temp_train_data = temp_train_data.sort_values(['ImageId', 'ClassId'],ascending=[True,True])
train_data = temp_train_data.pivot_table('EncodedPixels',['ImageId'], 'ClassId', aggfunc = np.sum).astype(str)

#
#encoded_pixels = train_data['EncodedPixels']
train_data = train_data.reset_index()
train_data = train_data.replace('nan',np.nan)
train_data = train_data.fillna(0)
train_data.columns = ['ImageId', 'D0','D1','D2','D3','D4']
train_data['H1'] = train_data.apply(lambda x: 0 if (x['D1']==0) else 1, axis = 1)
train_data['H2'] = train_data.apply(lambda x: 0 if (x['D2']==0) else 1, axis = 1)
train_data['H3'] = train_data.apply(lambda x: 0 if (x['D3']==0) else 1, axis = 1)
train_data['H4'] = train_data.apply(lambda x: 0 if (x['D4']==0) else 1, axis = 1)
train_data['Defect'] = train_data.apply(lambda x: 1 if ((x['H1'] or x['H2'] or x['H3'] or x['H4'])==1) else 0, axis = 1)
train_data['no_of_defect'] = train_data.apply(lambda x: x.H1+x.H2+x.H3+x.H4, axis = 1)
#train_data['no_of_defect'].value_counts().sort_index(ascending=True).plot('bar')
Y_train_bin = train_data['Defect']
#Y_train_multi = train_data[['H1','H2','H3','H4']][train_data['Defect']==1]

#X_train_multi = np.zeros((len(Y_train_multi),256,512,3),dtype=np.uint8)

#train_data['powerlabel'] = train_data.apply(lambda x : 8*x['H4']+4*x['H3']+2*x['H2']+1*x['H1'],axis=1)
train_data = train_data[['ImageId','D1','D2','D3','D4']][train_data['Defect']==1]

print(train_data.head(10))
rmv_idx = []
X_train = np.zeros((len(train_data)*4,256,384,3),dtype=np.uint8)
Y_train = np.zeros((len(train_data)*4,256,384,1),dtype=np.uint8)
#plt.show()
j = 0
defect = 0
for idx, i in enumerate(train_data.values,0):
    temp_X = cv2.imread(train_path + i[0],1)
    mask = np.zeros((256,1600),dtype=np.uint8)
    defect = 0
    if(i[1]!=0):
        temp_Y = rle2mask(i[1],1)
        mask = np.maximum(mask,temp_Y)
        #print(mask)
        defect = defect + 1
    if(i[2]!=0):
        temp_Y = rle2mask(i[2],2)
        mask = np.maximum(mask,temp_Y)
        defect = defect + 1
    if(i[3]!=0):
        #temp_Y = np.reshape(rle2mask(i[3],3),(256,1600,-1))
        #print('3')
        temp_Y = rle2mask(i[3],3)
        mask = np.maximum(mask,temp_Y)
        defect = defect + 1
    if(i[4]!=0):
        #temp_Y = np.reshape(rle2mask(i[4],4),(256,1600,-1))
        #print('4')
        temp_Y = rle2mask(i[4],4)
        mask = np.maximum(mask,temp_Y)
        defect = defect + 1
    temp_Y = cv2.resize(mask,(1600,256),interpolation = cv2.INTER_NEAREST)
    #temp_Y = np.reshape(temp_Y,(256,1600,-1))
    

    temp_img = cv2.resize(temp_X[:,:400,:],(384,256),interpolation = cv2.INTER_NEAREST)
    X_train[j] = temp_img
    temp_img = cv2.resize(temp_Y[:,:400],(384,256),interpolation = cv2.INTER_NEAREST)
    temp_img = np.reshape(temp_img,(256,384,-1))
    Y_train[j] = temp_img
    if(np.all(Y_train[j]==0)):
        rmv_idx.append(j)
    j = j + 1

    temp_img = cv2.resize(temp_X[:,400:800,:],(384,256),interpolation = cv2.INTER_NEAREST)
    X_train[j] = temp_img
    temp_img = cv2.resize(temp_Y[:,400:800],(384,256),interpolation = cv2.INTER_NEAREST)
    temp_img = np.reshape(temp_img,(256,384,-1))
    Y_train[j] = temp_img
    if(np.all(Y_train[j]==0)):
        rmv_idx.append(j)
    j = j + 1

    temp_img = cv2.resize(temp_X[:,800:1200,:],(384,256),interpolation = cv2.INTER_NEAREST)
    X_train[j] = temp_img
    temp_img = cv2.resize(temp_Y[:,800:1200],(384,256),interpolation = cv2.INTER_NEAREST)
    temp_img = np.reshape(temp_img,(256,384,-1))
    Y_train[j] = temp_img
    if(np.all(Y_train[j]==0)):
        rmv_idx.append(j)
    j = j + 1

    temp_img = cv2.resize(temp_X[:,1200:,:],(384,256),interpolation = cv2.INTER_NEAREST)
    X_train[j] = temp_img
    temp_img = cv2.resize(temp_Y[:,1200:],(384,256),interpolation = cv2.INTER_NEAREST)
    temp_img = np.reshape(temp_img,(256,384,-1))
    Y_train[j] = temp_img
    if(np.all(Y_train[j]==0)):
        rmv_idx.append(j)
    j = j + 1
SEED = 124
combine = [ ]
X_train = np.delete(X_train,rmv_idx,0)
Y_train = np.delete(Y_train,rmv_idx,0)
print(Y_train[0])
print(Y_train.shape)



x_tra, x_tst, y_tra, y_tst = train_test_split(X_train, Y_train, test_size = 0.1, shuffle=True,random_state=100)
x_tra, x_val, y_tra, y_val = train_test_split(x_tra, y_tra, test_size = 0.1,  shuffle=True,random_state=100)
X_train = []
Y_train = []

epochs = 30
learning_rate = 0.0001 
decay_rate = 0.0001
momentum = 0.8
eta_max=2e-3
eta_min=1e-5
sgd = SGD(lr=0.1, momentum=0.9)
SEED = 123

callbacks = [CosineAnnealingScheduler(T_max=5, eta_max=0.0005, eta_min=0.00025)]

Aug = albumentations.Compose([
                 albumentations.OneOf([        
                 albumentations.HorizontalFlip(p=0.2), 
                 albumentations.GridDistortion(num_steps=5, distort_limit=(0,0.03),p=0.2),
                 albumentations.VerticalFlip(p=0.2),
                 albumentations.RandomResizedCrop(256, 384, scale=(0.4, 0.8), p=0.2),
                 albumentations.OpticalDistortion(distort_limit=0.03, shift_limit=0.03,p=0.2),
                 albumentations.RandomResizedCrop(256, 384, scale=(0.4, 0.8), p=0.2)],p=1)])
                 #albumentations.GaussianBlur((0.0,3.0), p=0.5)],p=1)])
                #albumentations.MedianBlur(blur_limit=3, p=0.1)])],p=1)



training_datagen = ImageDataAugmentor(augment = Aug, augment_seed = SEED)
mask_datagen = ImageDataAugmentor(augment=Aug,augment_seed = SEED)
validation_training_datagen = ImageDataAugmentor()
validation_mask_datagen = ImageDataAugmentor()
testing_datagen = ImageDataAugmentor()

true_y = y_tst.astype(np.uint8)
true_y = true_y.astype(np.float32)



image_data_augmentator = training_datagen.flow(x_tra, batch_size=8, shuffle=False)
mask_data_augmentator = mask_datagen.flow(y_tra,batch_size=8,shuffle=False)

val_image_data_augmentator = validation_training_datagen.flow(x_val, batch_size=8, shuffle=False)
val_mask_data_augmentator = validation_mask_datagen.flow(y_val,batch_size=8,shuffle=False)


training_data_generator = zip(image_data_augmentator, mask_data_augmentator)
val_training_data_generator = zip(val_image_data_augmentator, val_mask_data_augmentator)

from sklearn.utils import class_weight

def local_tversky(y_true, y_pred, smooth=1e-6, alpha=0.7):
    #y_true_pos = K.flatten(y_true)
    #y_pred_pos = K.flatten(y_pred)
    temp_true = to_categorical(y_true, num_classes=5)
    y_true_pos = (temp_true[...,1:]).flatten()
    y_pred_pos = (y_pred[...,1:]).flatten()
    true_pos = np.sum(y_true_pos * y_pred_pos,axis=-1)
    false_neg = np.sum(y_true_pos * (1 - y_pred_pos),axis=-1)
    false_pos = np.sum((1 - y_true_pos) * y_pred_pos,axis=-1)
    return (true_pos + smooth) / (true_pos + 0.7 * false_neg + 0.3 * false_pos + smooth)

def local_tversky_class(y_true, y_pred, i, smooth=1e-6, alpha=0.7):
    #y_true_pos = K.flatten(y_true)
    #y_pred_pos = K.flatten(y_pred)
 #y_pred_pos = K.flatten(y_pred)
    temp_true = to_categorical(y_true, num_classes=5)
    y_true_pos = (temp_true[...,i]).flatten()
    y_pred_pos = (y_pred[...,i]).flatten()
    true_pos = np.sum(y_true_pos * y_pred_pos,axis=-1)
    false_neg = np.sum(y_true_pos * (1 - y_pred_pos),axis=-1)
    false_pos = np.sum((1 - y_true_pos) * y_pred_pos,axis=-1)
    return (true_pos + smooth) / (true_pos + 0.7 * false_neg + 0.3 * false_pos + smooth)


temp_prec = []
temp_recall = []
temp_area = []
tverskyindex = []
class_name = ['class1', 'class2', 'class3', 'class4','Overall']




def categorical_focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        max_num = K.constant(256*384)
        y_true_pos = K.one_hot(K.cast(y_true, 'int32'), num_classes=5)
        y_true_pos = Lambda(lambda x: x[:, :, :,0, :])(y_true_pos)
        y_pred_pos = K.clip(y_pred[...,0:], epsilon, 1. - epsilon)
        w = K.sum(K.sum(y_true_pos,axis = 2),axis=1)
        w = K.clip(w, epsilon, max_num)
        w = 1/w
        w = w[:,None,None,:]
        cross_entropy = -y_true_pos * K.log(y_pred_pos)
        loss =  w*(K.pow(1 - y_pred_pos, 1) * cross_entropy)
        return K.mean(K.sum(loss, axis=-1))


model = sm.Unet('resnet34', input_shape = (None,None,3), classes=5, activation='softmax', encoder_weights='imagenet')

model.compile(optimizer=Adam(0.001),loss=categorical_focal_loss_fixed, metrics =['sparse_categorical_accuracy'])  

results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=30, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
model.save("steel_SCE(n)")
acc1 = results.history['sparse_categorical_accuracy']
loss1 = results.history['loss']
vacc1 = results.history['val_sparse_categorical_accuracy']
vloss1 = results.history['val_loss']


#testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, batch_size=4)


samples = len(x_tst)
y_preds = np.argmax(preds_train,axis=-1)
y_preds = np.expand_dims(y_preds, axis=-1)
y_preds = y_preds.astype(np.float32)
target_names = ['class 0', 'class 1','class 2','class 3', 'class 4']
print(classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names))
df1 = classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names,output_dict=True)
df1 = pd.DataFrame(df1)
filename = 'steel_SCE(n)_report.csv'
df1.to_csv(filename,index=False)
ImageSave(x_tst, y_tst, y_preds, 'SCE(n)')
testAcc = local_tversky(true_y, preds_train)
print('DSC: ', testAcc)
class1 = local_tversky_class(true_y, preds_train,1)
class2 = local_tversky_class(true_y, preds_train,2)
class3 = local_tversky_class(true_y, preds_train,3)
class4 = local_tversky_class(true_y, preds_train,4)
tverskyindex.append(class1)
tverskyindex.append(class2)
tverskyindex.append(class3)
tverskyindex.append(class4)
#tverskyindex.append(testAcc)
submission = pd.DataFrame({'Class': ['C1', 'C2','C3', 'C4'], 'DSC': tverskyindex})
filename = 'steel_DSC_SCE(n).csv'
submission.to_csv(filename,index=False)
tverskyindex = []



model = sm.Unet('resnet34', input_shape = (None,None,3), classes=5, activation='softmax', encoder_weights='imagenet')

model.compile(optimizer=Adam(0.001),loss=categorical_focal_loss_fixed, metrics =['sparse_categorical_accuracy'])  
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=30, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
model.save("steel_SCE(n) + FL(γ=1)")
acc2 = results.history['sparse_categorical_accuracy']
loss2 = results.history['loss']
vacc2 = results.history['val_sparse_categorical_accuracy']
vloss2 = results.history['val_loss']

#testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, batch_size=4)
samples = len(x_tst)
y_preds = np.argmax(preds_train,axis=-1)
y_preds = np.expand_dims(y_preds, axis=-1)
y_preds = y_preds.astype(np.float32)
target_names = ['class 0', 'class 1','class 2','class 3', 'class 4']
df2 = classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names,output_dict=True)
df2 = pd.DataFrame(df2)
filename = 'steel_SCE(n) + FL(γ=1)_report.csv'
df2.to_csv(filename,index=False)
ImageSave(x_tst, y_tst, y_preds, 'SCE(n) + FL(γ=1)')
testAcc = local_tversky(true_y, preds_train)
print('DSC: ', testAcc)
class1 = local_tversky_class(true_y, preds_train,1)
class2 = local_tversky_class(true_y, preds_train,2)
class3 = local_tversky_class(true_y, preds_train,3)
class4 = local_tversky_class(true_y, preds_train,4)
tverskyindex.append(class1)
tverskyindex.append(class2)
tverskyindex.append(class3)
tverskyindex.append(class4)
#tverskyindex.append(testAcc)
submission = pd.DataFrame({'Class': ['C1', 'C2','C3', 'C4'], 'DSC': tverskyindex})
filename = 'steel_DSC_(SCE(n) + FL(1)).csv'
submission.to_csv(filename,index=False)
tverskyindex = []


model = sm.Unet('resnet34', input_shape = (None,None,3), classes=5, activation='softmax', encoder_weights='imagenet')

model.compile(optimizer=Adam(0.001),loss=categorical_focal_loss_fixed, metrics =['sparse_categorical_accuracy'])  
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=30, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
model.save("steel_SCE(n) + FL(γ=2)")
acc3 = results.history['sparse_categorical_accuracy']
loss3 = results.history['loss']
vacc3 = results.history['val_sparse_categorical_accuracy']
vloss3 = results.history['val_loss']

#testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, batch_size=4)
samples = len(x_tst)
y_preds = np.argmax(preds_train,axis=-1)
y_preds = np.expand_dims(y_preds, axis=-1)
y_preds = y_preds.astype(np.float32)
target_names = ['class 0', 'class 1','class 2','class 3', 'class 4']
df3 = classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names,output_dict=True)
df3 = pd.DataFrame(df3)
filename = 'steel_SCE(n) + FL(γ=2)_report.csv'
df3.to_csv(filename,index=False)
ImageSave(x_tst, y_tst, y_preds, 'SCE(n) + FL(γ=2)')
testAcc = local_tversky(true_y, preds_train)
print('DSC: ', testAcc)
class1 = local_tversky_class(true_y, preds_train,1)
class2 = local_tversky_class(true_y, preds_train,2)
class3 = local_tversky_class(true_y, preds_train,3)
class4 = local_tversky_class(true_y, preds_train,4)
tverskyindex.append(class1)
tverskyindex.append(class2)
tverskyindex.append(class3)
tverskyindex.append(class4)
#tverskyindex.append(testAcc)
submission = pd.DataFrame({'Class': ['C1', 'C2','C3', 'C4'], 'DSC': tverskyindex})
filename = 'steel_DSC_(SCE(n) + FL(2)).csv'
submission.to_csv(filename,index=False)
tverskyindex = []
model = sm.Unet('resnet34', input_shape = (None,None,3), classes=5, activation='softmax', encoder_weights='imagenet')

model.compile(optimizer=Adam(0.001),loss=categorical_focal_loss_fixed, metrics =['sparse_categorical_accuracy'])  
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=30, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
model.save("steel_SCE(n) + FL(γ=3)")
acc4 = results.history['sparse_categorical_accuracy']
loss4 = results.history['loss']
vacc4 = results.history['val_sparse_categorical_accuracy']
vloss4 = results.history['val_loss']

#testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, batch_size=4)
samples = len(x_tst)
y_preds = np.argmax(preds_train,axis=-1)
y_preds = np.expand_dims(y_preds, axis=-1)
y_preds = y_preds.astype(np.float32)
target_names = ['class 0', 'class 1','class 2','class 3', 'class 4']
print(classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names))
df4 = classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names,output_dict=True)
df4 = pd.DataFrame(df4)
filename = 'steel_SCE(n) + FL(γ=3)_report.csv'
df4.to_csv(filename,index=False)
ImageSave(x_tst, y_tst, y_preds, 'SCE(n) + FL(γ=3)')
testAcc = local_tversky(true_y, preds_train)
print('DSC: ', testAcc)
class1 = local_tversky_class(true_y, preds_train,1)
class2 = local_tversky_class(true_y, preds_train,2)
class3 = local_tversky_class(true_y, preds_train,3)
class4 = local_tversky_class(true_y, preds_train,4)
tverskyindex.append(class1)
tverskyindex.append(class2)
tverskyindex.append(class3)
tverskyindex.append(class4)
#tverskyindex.append(testAcc)
submission = pd.DataFrame({'Class': ['C1', 'C2','C3', 'C4'], 'DSC': tverskyindex})
filename = 'steel_DSC_(SCE(n) + FL(3)).csv'
submission.to_csv(filename,index=False)
tverskyindex = []


model = sm.Unet('resnet34', input_shape = (None,None,3), classes=5, activation='softmax', encoder_weights='imagenet')

model.compile(optimizer=Adam(0.001),loss=categorical_focal_loss_fixed, metrics =['sparse_categorical_accuracy'])  
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=30, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
model.save("steel_SCE(n) + FL(γ=4)")
acc5 = results.history['sparse_categorical_accuracy']
loss5 = results.history['loss']
vacc5 = results.history['val_sparse_categorical_accuracy']
vloss5 = results.history['val_loss']

#testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, batch_size=4)
samples = len(x_tst)
y_preds = np.argmax(preds_train,axis=-1)
y_preds = np.expand_dims(y_preds, axis=-1)
y_preds = y_preds.astype(np.float32)
target_names = ['class 0', 'class 1','class 2','class 3', 'class 4']
print(classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names))
df5 = classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names,output_dict=True)
df5 = pd.DataFrame(df5)
filename = 'steel_SCE(n) + FL(γ=4)_report.csv'
df5.to_csv(filename,index=False)
ImageSave(x_tst, y_tst, y_preds, 'SCE(n) + FL(γ=4)')
testAcc = local_tversky(true_y, preds_train)
print('DSC: ', testAcc)
class1 = local_tversky_class(true_y, preds_train,1)
class2 = local_tversky_class(true_y, preds_train,2)
class3 = local_tversky_class(true_y, preds_train,3)
class4 = local_tversky_class(true_y, preds_train,4)
tverskyindex.append(class1)
tverskyindex.append(class2)
tverskyindex.append(class3)
tverskyindex.append(class4)
#tverskyindex.append(testAcc)
submission = pd.DataFrame({'Class': ['C1', 'C2','C3', 'C4'], 'DSC': tverskyindex})
filename = 'steel_DSC_(SCE(n) + FL(4)).csv'
submission.to_csv(filename,index=False)
tverskyindex = []



model = sm.Unet('resnet34', input_shape = (None,None,3), classes=5, activation='softmax', encoder_weights='imagenet')

model.compile(optimizer=Adam(0.001),loss=categorical_focal_loss_fixed, metrics =['sparse_categorical_accuracy'])  
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=30, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
model.save("steel_SCE(n) + FL(γ=5)")
acc6 = results.history['sparse_categorical_accuracy']
loss6 = results.history['loss']
vacc6 = results.history['val_sparse_categorical_accuracy']
vloss6 = results.history['val_loss']

#testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, batch_size=4)
samples = len(x_tst)
y_preds = np.argmax(preds_train,axis=-1)
y_preds = np.expand_dims(y_preds, axis=-1)
y_preds = y_preds.astype(np.float32)
target_names = ['class 0', 'class 1','class 2','class 3', 'class 4']
print(classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names))
df6 = classification_report(true_y.flatten(), y_preds.flatten(), target_names=target_names,output_dict=True)
df6 = pd.DataFrame(df6)
filename = 'steel_SCE(n) + FL(γ=5)_report.csv'
df6.to_csv(filename,index=False)
ImageSave(x_tst, y_tst, y_preds, 'SCE(n) + FL(γ=5)')
testAcc = local_tversky(true_y, preds_train)
print('DSC: ', testAcc)
class1 = local_tversky_class(true_y, preds_train,1)
class2 = local_tversky_class(true_y, preds_train,2)
class3 = local_tversky_class(true_y, preds_train,3)
class4 = local_tversky_class(true_y, preds_train,4)
tverskyindex.append(class1)
tverskyindex.append(class2)
tverskyindex.append(class3)
tverskyindex.append(class4)
#tverskyindex.append(testAcc)
submission = pd.DataFrame({'Class': ['C1', 'C2','C3', 'C4'], 'DSC': tverskyindex})
filename = 'steel_DSC_(SCE(n) + FL(5)).csv'
submission.to_csv(filename,index=False)
tverskyindex = []




fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(acc1, 'r', label='SCE')
ax.plot(acc2, 'b', label='SCE + FL(γ=1)')
ax.plot(acc3, 'g', label='SCE + FL(γ=2)')
ax.plot(acc4, 'y', label='SCE + FL(γ=3)')
ax.plot(acc5, 'c', label='SCE + FL(γ=4)')
ax.plot(acc6, 'm', label='SCE + FL(γ=5)')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_all_acc_SCE(n).png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(loss1, 'r', label='SCE')
ax.plot(loss2, 'b', label='SCE + FL(γ=1)')
ax.plot(loss3, 'g', label='SCE + FL(γ=2)')
ax.plot(loss4, 'y', label='SCE + FL(γ=3)')
ax.plot(loss5, 'c', label='SCE + FL(γ=4)')
ax.plot(loss6, 'm', label='SCE + FL(γ=5)')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_all_loss_SCE(n).png')
plt.clf()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vacc1, 'r', label='SCE')
ax.plot(vacc2, 'b', label='SCE + FL(γ=1)')
ax.plot(vacc3, 'g', label='SCE + FL(γ=2)')
ax.plot(vacc4, 'y', label='SCE + FL(γ=3)')
ax.plot(vacc5, 'c', label='SCE + FL(γ=4)')
ax.plot(vacc6, 'm', label='SCE + FL(γ=5)')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_all_vacc_SCE(n).png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vloss1, 'r', label='SCE')
ax.plot(vloss2, 'b', label='SCE + FL(γ=1)')
ax.plot(vloss3, 'g', label='SCE + FL(γ=2)')
ax.plot(vloss4, 'y', label='SCE + FL(γ=3)')
ax.plot(vloss5, 'c', label='SCE + FL(γ=4)')
ax.plot(vloss6, 'm', label='SCE + FL(γ=5)')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_all_vloss_SCE(n).png')
plt.clf()
