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
from sklearn.metrics import classification_report
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

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import , Dense, BatchNormalization, Dropout
seed = 42

def my_metrics(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    true_neg = K.sum(false_pos * false_neg)
    recall = true_pos/(true_pos+false_neg)
    precision = true_pos/(true_pos+false_pos)
    f1score = (2*recall*precision)/(precision+recall)
    return recall, precision, f1score
    #cm=confusion_matrix(y_true, y_pred)
def ImageSave(x_tst, true_y, preds_train, aug_type):
    for i in range(3):
        plt.subplot(3, 3, 1 + i)
        plt.axis('off')
        plt.imshow(x_tst[i].astype(np.int32))
        #plt.imshow(x_tst[i],cmap='gray')
    for i in range(3): 
        plt.subplot(3, 3, 1 + 3 + i)
        plt.axis('off')
        plt.imshow(true_y[i],cmap='gray')
    # plot real target image
    for i in range(3):
        plt.subplot(3, 3, 1 + 3*2 + i)
        plt.axis('off')
        plt.imshow(preds_train[i],cmap='gray')
    filename1 = 't1_' + aug_type + '1' + '.png'
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
        plt.imshow(true_y[i],cmap='gray')
        idx = idx+1
    # plot real target image
    idx = 0
    for i in range(4,7):
        plt.subplot(3, 3, 1 + 3*2 + idx)
        plt.axis('off')
        plt.imshow(preds_train[i],cmap='gray')
        idx = idx+1
    filename1 = 't1_' + aug_type + '2' + '.png'
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
        plt.imshow(true_y[i],cmap='gray')
        idx = idx+1
    # plot real target image
    idx = 0
    for i in range(8,11):
        plt.subplot(3, 3, 1 + 3*2 + idx)
        plt.axis('off')
        plt.imshow(preds_train[i],cmap='gray')
        idx = idx+1
    filename1 = 't1_' + aug_type + '3' + '.png'
    plt.savefig(filename1)
    plt.close()
def rle2mask(mask_rle, shape=(1600,256)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype =np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def tversky(y_true, y_pred, smooth=1e-6, alpha=0.9):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + 0.9 * false_neg + 0.1 * false_pos + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 1)

ALPHA = 0.25
GAMMA = 2

'''def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss'''

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score




def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0],pixels,[0]])
    runs = np.where(pixels[1:]!=pixels[:-1])[0]+1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def dice_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def big_unet():
    unet1 = sm.Unet('resnet34', classes = 1, activation='sigmoid', encoder_weights='imagenet')
    features_3 = unet1.get_layer('decoder_stage2b_relu').output
    print(features_3.shape)
    features_4 = unet1.get_layer('decoder_stage3b_relu').output
    print(features_4.shape)
    features_5 = unet1.get_layer('decoder_stage4b_relu').output
    print(features_5.shape)
    features_3 = UpSampling2D(size=(4,4))(features_3)
    features_4 = UpSampling2D(size=(2,2))(features_4)

    '''print(features_1.shape)
    print((unet1.get_layer('decoder_stage0_upsampling').output).shape)
    print((unet1.get_layer('decoder_stage0_concat').output).shape)

    print(features_2.shape)
    print((unet1.get_layer('decoder_stage1_upsampling').output).shape)
    print((unet1.get_layer('decoder_stage1_concat').output).shape)

    print(features_3.shape)
    print((unet1.get_layer('decoder_stage2_upsampling').output).shape)
    print((unet1.get_layer('decoder_stage2_concat').output).shape)

    print(features_4.shape)
    print((unet1.get_layer('decoder_stage3_upsampling').output).shape)
    print((unet1.get_layer('decoder_stage3_concat').output).shape)'''

    x = concatenate([features_5, features_4,features_3])
    x = Conv2D(
        filters=8,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='he_uniform',
        name='final_1',
        )(x)
    x = BatchNormalization()(x)
    result1 = Activation('relu')(x)
    unet1 = Model(input=unet1.input,output=result1)
    '''inp = Input(shape=(None,None,1))
    l1 = Conv2D(3,(1,1))(inp)
    out = unet1(l1)
    unet1 = Model(inp,out)'''
    unet1.summary()

 
    '''unet2 = sm.Unet('inceptionv3', classes = 1, activation='sigmoid', encoder_weights='imagenet')
    unet2.summary()
    features_1 = unet2.get_layer('decoder_stage0_upsampling').output
    print(features_1.shape)
    features_2 = unet2.get_layer('decoder_stage1_upsampling').output
    print(features_2.shape)
    features_3 = unet2.get_layer('decoder_stage2_upsampling').output
    print(features_3.shape)
    features_4 = unet2.get_layer('decoder_stage3_upsampling').output
    print(features_4.shape)
    features_5 = unet2.get_layer('decoder_stage4_upsampling').output
    print(features_5.shape)
    features_1 = UpSampling2D(size=(16,16))(features_1)
    features_2 = UpSampling2D(size=(8,8))(features_2)
    features_3 = UpSampling2D(size=(4,4))(features_3)
    features_4 = UpSampling2D(size=(2,2))(features_4)
    x = concatenate([features_5, features_4,features_3])
    x = Conv2D(
        filters=4,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='he_uniform',
        name='final_3',
        )(x)
    x = BatchNormalization()(x)
    result2 = Activation('relu')(x)'''
    unet2 = sm.FPN('resnet34', classes = 1, activation='sigmoid', encoder_weights='imagenet')
    unet2.summary()
    final = unet2.get_layer('final_stage_relu').output
    final = UpSampling2D(size=(2,2))(final)
    '''final = Conv2D(
        filters=8,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='he_uniform',
        name='final_2',
        )(final)'''
    unet2 = Model(input=unet2.input,output=final)
    '''inp = Input(shape=(None,None,1))
    l1 = Conv2D(3,(1,1))(inp)
    out = unet2(l1)
    unet2 = Model(inp,out)'''

    for layer in unet2.layers:
        layer.name = layer.name + str("_2")

    dual_unet = Model(input=[unet1.input,unet2.input],output=[unet1.output,unet2.output])

    final = concatenate([dual_unet.output[0], dual_unet.output[1]])
    final_conv = Conv2D(
        filters=8,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='he_uniform',
        name='final_conv1',
        )(final)
    final = BatchNormalization()(final_conv)
    final = Activation('relu')(final)
    final = Conv2D(
        filters=8,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='he_uniform',
        name='final_conv2',
        )(final)
    final = BatchNormalization()(final)
    final = Activation('relu')(final)
    final_conv = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final',
        )(final)
    final_result = Activation('sigmoid')(final_conv)
    final_unet = Model(input=dual_unet.input,output=final_result)
    return final_unet

train_path = 'severstal-steel-defect-detection/train_images/'
test_path = 'severstal-steel-defect-detection/test_images/'

train_images = next(os.walk(train_path))[2]
train_images.sort()
test_images = next(os.walk(test_path))[2]
X_train = np.zeros((len(train_images),256,512,3),dtype=np.uint8)
X_test = np.zeros((len(test_images),256,512,3),dtype=np.uint8)
cnt = 0
#train_path = 'severstal-steel-defect-detection/train_images_2/'

train_images_df = pd.DataFrame(train_images,columns =['ImageId'])
train_mask_df = pd.read_csv("severstal-steel-defect-detection/train.csv")
temp_train_data = train_mask_df.merge(train_images_df, how='outer',on='ImageId')
temp_train_data['ClassId'] = temp_train_data['ClassId'].fillna(0)
'''plot = temp_train_data['ClassId'].value_counts().sort_index().plot.bar()
plt.xlabel('ClassId')
plt.ylabel('No. of Images')
plt.show()'''
#temp_train_data['ClassId'].value_counts().sort_index(ascending=True).plot('bar')
#plt.show()
temp_train_data = temp_train_data.sort_values(['ImageId', 'ClassId'],ascending=[True,True])
train_data = temp_train_data.pivot_table('EncodedPixels',['ImageId'], 'ClassId', aggfunc = np.sum).astype(str)

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
Y_train_bin = train_data['Defect']

train_data['powerlabel'] = train_data.apply(lambda x : 8*x['H4']+4*x['H3']+2*x['H2']+1*x['H1'],axis=1)
Y_train_multi = train_data['powerlabel']
Y_train_multi = to_categorical(Y_train_multi)
print((train_data['H1']==1).sum())
print((train_data['H2']==1).sum())
print((train_data['H3']==1).sum())
print((train_data['H4']==1).sum())


#plt.show()


train_data_type1 = train_data[['ImageId','D1']][train_data['H1']==1]
t1_X_train = np.zeros((len(train_data_type1),256,1600,3),dtype=np.uint8)
t1_Y_train = np.zeros((len(train_data_type1),256,1600,1),dtype=np.uint8)


image_id=[]
t1_defect = []
t1_defect_ori = []
from skimage import filters
for idx, i in enumerate(train_data_type1.values,0):
    temp_X = cv2.imread(train_path + i[0],0)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))
    temp_X = clahe.apply(temp_X)
    temp_X = cv2.cvtColor(temp_X, cv2.COLOR_GRAY2BGR)
    t1_X_train[idx] = temp_X
    t1_defect_ori.append((rle2mask(i[1]) == 1).sum())
    temp_Y = rle2mask(i[1])
    t1_Y_train[idx] = np.reshape(temp_Y,(256,1600,-1))
    t1_defect.append((t1_Y_train[idx] == 1).sum())





rmv_idx = []
'''for idx, item in enumerate(t1_Y_train,0):
    if((((item == 1).sum())>np.percentile(t1_defect, 98)) or (((item == 1).sum())<np. percentile(t1_defect, 2))):
        rmv_idx.append(idx)'''

t1_defect = []
#t1_X_train = np.delete(t1_X_train,rmv_idx,0)
#t1_Y_train = np.delete(t1_Y_train,rmv_idx,0)


new_X_train = np.zeros((len(t1_X_train)*4,256,384,3),dtype=np.uint8)
new_Y_train = np.zeros((len(t1_Y_train)*4,256,384,1),dtype=np.uint8)
rmv_idx = []
i = 0
print(len(new_X_train))
print(len(new_Y_train))
for a , b in zip(t1_X_train,t1_Y_train):
    temp_X = a[:,:400,:]
    temp_img = cv2.resize(temp_X,(384,256),interpolation = cv2.INTER_LINEAR)
    #temp_img = np.reshape(temp_img,(256,384,-1))
    new_X_train[i] = temp_img
    temp_Y = b[:,:400]
    temp_img = cv2.resize(temp_Y,(384,256),interpolation = cv2.INTER_LINEAR)
    temp_img = np.reshape(temp_img,(256,384,-1))
    new_Y_train[i] = temp_img
    if(np.all(new_Y_train[i]==0)):
        rmv_idx.append(i)
    else:
        t1_defect.append((new_Y_train[i] == 1).sum())
    i = i + 1
    temp_X = a[:,400:800,:]
    temp_img = cv2.resize(temp_X,(384,256),interpolation = cv2.INTER_LINEAR)
    #temp_img = np.reshape(temp_img,(256,384,-1))
    new_X_train[i] = temp_img
    temp_Y = b[:,400:800]
    temp_img = cv2.resize(temp_Y,(384,256),interpolation = cv2.INTER_LINEAR)
    temp_img = np.reshape(temp_img,(256,384,-1))
    new_Y_train[i] = temp_img
    if(np.all(new_Y_train[i]==0)):
        rmv_idx.append(i)
    else:
        t1_defect.append((new_Y_train[i] == 1).sum())
    i = i + 1
    temp_X = a[:,800:1200,:]
    temp_img = cv2.resize(temp_X,(384,256),interpolation = cv2.INTER_LINEAR)
    #temp_img = np.reshape(temp_img,(256,384,-1))
    new_X_train[i] = temp_img
    temp_Y = b[:,800:1200]
    temp_img = cv2.resize(temp_Y,(384,256),interpolation = cv2.INTER_LINEAR)
    temp_img = np.reshape(temp_img,(256,384,-1))
    new_Y_train[i] = temp_img
    if(np.all(new_Y_train[i]==0)):
        rmv_idx.append(i)
    else:
        t1_defect.append((new_Y_train[i] == 1).sum())
    i = i + 1
    temp_X = a[:,1200:,:]
    temp_img = cv2.resize(temp_X,(384,256),interpolation = cv2.INTER_LINEAR)
    #temp_img = np.reshape(temp_img,(256,384,-1))
    new_X_train[i] = temp_img
    temp_Y = b[:,1200:]
    temp_img = cv2.resize(temp_Y,(384,256),interpolation = cv2.INTER_LINEAR)
    temp_img = np.reshape(temp_img,(256,384,-1))
    new_Y_train[i] = temp_img
    if(np.all(new_Y_train[i]==0)):
        rmv_idx.append(i)
    else:
        t1_defect.append((new_Y_train[i] == 1).sum())
    i = i + 1
    '''new_X_train[i] = a[:,:800,:]
    new_Y_train[i] = b[:,:800]
    if(np.all(new_Y_train[i]==0)):
        rmv_idx.append(i)
    i = i + 1
    new_X_train[i] = a[:,800:1600,:]
    new_Y_train[i] = b[:,800:1600]
    if(np.all(new_Y_train[i]==0)):
        rmv_idx.append(i)
    i = i + 1'''
new_X_train = np.delete(new_X_train,rmv_idx,0)
new_Y_train = np.delete(new_Y_train,rmv_idx,0)
rmv_idx = []

for idx, item in enumerate(new_Y_train,0):
    if((((item == 1).sum())>np.percentile(t1_defect, 98)) or (((item == 1).sum())<np. percentile(t1_defect, 2))):
        rmv_idx.append(idx)
new_X_train = np.delete(new_X_train,rmv_idx,0)
new_Y_train = np.delete(new_Y_train,rmv_idx,0)
t1_X_train = []
t1_Y_train = []
SEED = 123


x_tra, x_val, y_tra, y_val = train_test_split(new_X_train, new_Y_train, test_size = 0.1, shuffle=True, random_state = 42)

x_tra, x_tst, y_tra, y_tst = train_test_split(x_tra, y_tra, test_size = 0.1,  shuffle=True, random_state = 42)

true_y = y_tst.astype(np.uint8)
true_y = true_y.astype(np.float32)
print(x_tra.shape)
sgd = SGD(lr=0.1, momentum=0.9)
callbacks = [
    CosineAnnealingScheduler(T_max=5, eta_max=0.0005, eta_min=0.00025)
]

print(new_X_train.shape)

Aug = albumentations.Compose([
                 albumentations.OneOf([   
                 albumentations.HorizontalFlip(p=0.5), 
                 albumentations.GridDistortion(num_steps=5, distort_limit=(0,0.03),p=0.2),
                 albumentations.VerticalFlip(p=0.5),
                 albumentations.RandomResizedCrop(256, 384, scale=(0.4, 0.8), p=0.2),
                 albumentations.OpticalDistortion(distort_limit=0.03, shift_limit=0.03,p=0.2)],p=1)])

training_datagen = ImageDataAugmentor(augment = Aug, augment_seed = SEED)
mask_datagen = ImageDataAugmentor(augment=Aug,augment_seed = SEED)
validation_training_datagen = ImageDataAugmentor()
validation_mask_datagen = ImageDataAugmentor()

testing_datagen = ImageDataAugmentor()

image_data_augmentator = training_datagen.flow(x_tra, batch_size=8, shuffle=False)
mask_data_augmentator = mask_datagen.flow(y_tra,batch_size=8,shuffle=False)

val_image_data_augmentator = validation_training_datagen.flow(x_val, batch_size=8, shuffle=False)
val_mask_data_augmentator = validation_mask_datagen.flow(y_val,batch_size=8,shuffle=False)


training_data_generator = zip(image_data_augmentator, mask_data_augmentator)
val_training_data_generator = zip(val_image_data_augmentator, val_mask_data_augmentator)


def generator_two_img(X, Y):
    image_data_augmentator_1 = training_datagen.flow(X, batch_size=8, shuffle=False)
    mask_data_augmentator_1 = mask_datagen.flow(Y,batch_size=8,shuffle=False)
    image_data_augmentator_2 = training_datagen.flow(X, batch_size=8, shuffle=False)
    mask_data_augmentator_2 = mask_datagen.flow(Y,batch_size=8,shuffle=False)
    while True:
        X1i = image_data_augmentator_1.next()
        X1j = mask_data_augmentator_1.next()
        yield [X1i, X1i], X1j
gen_flow = generator_two_img(x_tra, y_tra)

def generator_two_img(X, Y):
    image_data_augmentator_1 = validation_training_datagen.flow(X, batch_size=8, shuffle=False)
    mask_data_augmentator_1 = validation_mask_datagen.flow(Y,batch_size=8,shuffle=False)
    while True:
        X1i = image_data_augmentator_1.next()
        X1j = mask_data_augmentator_1.next()
        yield [X1i, X1i], X1j
val_flow = generator_two_img(x_val, y_val)

def test_two_img(X):
    image_data_augmentator_1 = testing_datagen.flow(X, batch_size=8, shuffle=False)
    while True:
        X1i = image_data_augmentator_1.next()
        X2i = image_data_augmentator_1.next()
        yield [X1i, X2i]
test_flow = test_two_img(x_tst)

model2 = big_unet()

model2.summary()


model2.compile(optimizer=AccumOptimizer(Adam(0.001),2),loss=focal_tversky_loss, metrics = [tversky])
results = model2.fit_generator(gen_flow, steps_per_epoch=len(x_tra)//8, epochs=60,validation_data=val_flow, validation_steps=len(x_val)//8)
model2.save("steel_t1_big")
preds_train = model2.predict([x_tst,x_tst],batch_size=4)
preds_train = (preds_train > 0.5).astype(np.float32)
precision, recall, _ = precision_recall_curve(true_y.flatten(), preds_train.flatten())
area = auc(recall,precision)
plt.plot(recall, precision,label='AUC = %.3f' %(area))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('PR Curve')
plt.savefig('steel_t1_PR_BigUnet.png')
plt.clf()
acc1 = results.history['tversky']
loss1 = results.history['loss']
vacc1 = results.history['val_tversky']
vloss1 = results.history['val_loss']
from sklearn.metrics import roc_curve, roc_auc_score
target_names = ['class 0', 'class 1']
testAcc = K.get_session().run(tversky(true_y, preds_train))
print('DSC: ', testAcc)
df1 = classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names,output_dict=True)
df1 = pd.DataFrame(df1)
ImageSave(x_tst, true_y, preds_train, 'BigUnet')
'''from sklearn.metrics import roc_curve, roc_auc_score
target_names = ['class 0', 'class 1']
testAcc = K.get_session().run(tversky(true_y, preds_train))
print('DSC: ', testAcc)
df1 = classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names,output_dict=True)
df1 = pd.DataFrame(df1)
ImageSave(x_tst, true_y, preds_train, 'BigUnet')'''


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(acc1, 'r', label='Train')
ax.plot(vacc1, 'b', label='Validation')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_t1_acc_BigUnet.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(loss1, 'r', label='Train')
ax.plot(vloss1, 'b', label='Validation')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_t1_loss_BigUnet.png')
plt.clf()

filename = 'steel_t1_report_BigUnet.csv'
df1.to_csv(filename,index=False)
