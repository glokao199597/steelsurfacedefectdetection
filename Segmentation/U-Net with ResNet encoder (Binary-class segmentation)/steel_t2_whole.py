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

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import , Dense, BatchNormalization, Dropout
seed = 42

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
    filename1 = 't2_' + aug_type + '1' + '.png'
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
    filename1 = 't2_' + aug_type + '2' + '.png'
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
    filename1 = 't2_' + aug_type + '3' + '.png'
    plt.savefig(filename1)
    plt.close()


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
    y_pred_pos = K.argmax(y_pred,axis=-1)
    y_pred_pos = K.cast(y_pred_pos,'float32')
    y_pred_pos = K.flatten(y_pred_pos)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + 0.9 * false_neg + 0.1 * false_pos + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 1)

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
#print((train_data['ClassId']==0).sum())
#print(((train_data['ClassId']==1) or (train_data['ClassId']==2) or (train_data['ClassId']==3) or (train_data['ClassId']==4)).sum())
#Y_train_multi = train_data[['H1','H2','H3','H4']][train_data['Defect']==1]

#X_train_multi = np.zeros((len(Y_train_multi),256,512,3),dtype=np.uint8)

train_data['powerlabel'] = train_data.apply(lambda x : 8*x['H4']+4*x['H3']+2*x['H2']+1*x['H1'],axis=1)
Y_train_multi = train_data['powerlabel']
Y_train_multi = to_categorical(Y_train_multi)
print((train_data['H1']==1).sum())
print((train_data['H2']==1).sum())
print((train_data['H3']==1).sum())
print((train_data['H4']==1).sum())


#plt.show()


train_data_type1 = train_data[['ImageId','D2']][train_data['H2']==1]

#train_data_type1= train_data_type1.rename(columns={"D1": "EncodedPixels"})

#train_data_type4=train_data_type4.rename(columns={"D4": "EncodedPixels"})

#train = pd.concat([train_data_type1,train_data_type2,train_data_type3,train_data_type])
#train = train.fillna(0)


t2_X_train = np.zeros((len(train_data_type1),256,1600,3),dtype=np.uint8)
t2_Y_train = np.zeros((len(train_data_type1),256,1600,1),dtype=np.uint8)


image_id=[]
t2_defect = []
t2_defect_ori = []
from skimage import filters
for idx, i in enumerate(train_data_type1.values,0):
    temp_X = cv2.imread(train_path + i[0],0)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #gray_image = cv2.cvtColor(temp_X, cv2.COLOR_BGR2GRAY)
    #print(gray_image)
    bilateral = cv2.bilateralFilter(temp_X, 3, 150, 300)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    temp_X = clahe.apply(bilateral)
    temp_X = cv2.cvtColor(temp_X, cv2.COLOR_GRAY2RGB)
    #temp_X = cv2.cvtColor(temp_X, cv2.COLOR_GRAY2RGB)
    #bilateral = clahe.apply(bilateral)
    #temp_X = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2RGB)
    t2_X_train[idx] = temp_X
    t2_defect_ori.append((rle2mask(i[1]) == 1).sum())
    temp_Y = rle2mask(i[1])
    t2_Y_train[idx] = np.reshape(temp_Y,(256,1600,-1))
    t2_defect.append((t2_Y_train[idx] == 1).sum())
    '''plt.subplot(3,1,1)
    plt.imshow(bilateral,cmap='gray')
    plt.subplot(3,1,2)
    plt.imshow(temp_X,cmap='gray')
    bilateral = clahe.apply(bilateral)
    plt.subplot(3,1,3)
    plt.imshow(bilateral,cmap='gray')
    plt.show()'''
'''for b in new_Y_train:
    t2_defect.append((b == 1).sum())

rmv_idx = []
for idx, item in enumerate(new_Y_train,0):
    if((((item == 1).sum())>np.percentile(t2_defect, 98)) or (((item == 1).sum())<np. percentile(t2_defect, 2))):
        rmv_idx.append(idx)'''

rmv_idx = []
'''for idx, item in enumerate(t2_Y_train,0):
    if((((item == 1).sum())>np.percentile(t2_defect, 98)) or (((item == 1).sum())<np. percentile(t2_defect, 2))):
        rmv_idx.append(idx)'''

t2_defect = []
#t2_X_train = np.delete(t2_X_train,rmv_idx,0)
#t2_Y_train = np.delete(t2_Y_train,rmv_idx,0)


new_X_train = np.zeros((len(t2_X_train)*4,256,384,3),dtype=np.uint8)
new_Y_train = np.zeros((len(t2_Y_train)*4,256,384,1),dtype=np.uint8)
rmv_idx = []
i = 0
t2_defect_full = []
print(len(new_X_train))
print(len(new_Y_train))
for a , b in zip(t2_X_train,t2_Y_train):
    t2_defect_full.append((b==1).sum())
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
        t2_defect.append((new_Y_train[i] == 1).sum())
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
        t2_defect.append((new_Y_train[i] == 1).sum())
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
        t2_defect.append((new_Y_train[i] == 1).sum())
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
        t2_defect.append((new_Y_train[i] == 1).sum())
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
    if((((item == 1).sum())>np.percentile(t2_defect, 98)) or (((item == 1).sum())<np. percentile(t2_defect, 2))):
        rmv_idx.append(idx)

new_X_train = np.delete(new_X_train,rmv_idx,0)
new_Y_train = np.delete(new_Y_train,rmv_idx,0)
t2_X_train = []
t2_Y_train = []
SEED = 123


x_tra, x_val, y_tra, y_val = train_test_split(new_X_train, new_Y_train, test_size = 0.1, shuffle=True, random_state = 42)

x_tra, x_tst, y_tra, y_tst = train_test_split(x_tra, y_tra, test_size = 0.1,  shuffle=True, random_state = 42)

true_y = y_tst.astype(np.uint8)
true_y = true_y.astype(np.float32)

sgd = SGD(lr=0.1, momentum=0.9)
callbacks = [
    CosineAnnealingScheduler(T_max=10, eta_max=0.0005, eta_min=0.00025)
]

print(new_X_train.shape)

Aug = albumentations.Compose([
                 albumentations.OneOf([        
                 albumentations.HorizontalFlip(p=0.2), 
                 albumentations.GridDistortion(num_steps=5, distort_limit=(0,0.03),p=0.2),
                 albumentations.VerticalFlip(p=0.2),
                 albumentations.RandomResizedCrop(256, 384, scale=(0.4, 0.8), p=0.2),
                 albumentations.OpticalDistortion(distort_limit=0.03, shift_limit=0.03,p=0.2)],p=1)])
                 #albumentations.GaussianBlur((0.0,3.0), p=0.5)],p=1)])

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
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_tra),
                                                 y_tra.flatten())
model = sm.Unet('resnet34', classes = 1, activation='sigmoid', encoder_weights='imagenet')
#inp = Input(shape=(None,None,1))
#l1 = Conv2D(3,(1,1))(inp)
#out = model(l1)
#model = Model(inp,out)

def focal_loss(y_true,y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    BCE = K.binary_crossentropy(y_true_pos,y_pred_pos)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(0.75*(K.pow((1-BCE_EXP), 2)) * BCE)
    return focal_loss

def FL_TL(y_true,y_pred):
    fl = focal_loss(y_true,y_pred)
    tv = tversky(y_true, y_pred)
    tv_loss = 1 - tv
    loss = fl + tv_loss
    return loss


def BCE_Tversky(y_true,y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    BCE = K.mean(K.binary_crossentropy(y_true_pos,y_pred_pos))
    tv = tversky(y_true, y_pred)
    tv_floss = K.pow((1 - tv), 0.75)
    tv_loss = 1 - tv
    dice_loss = 1 - dsc(y_true, y_pred) 
    loss = BCE + tv_loss
    return loss


def FL_DL(y_true,y_pred):
    #y_pred_pos = K.flatten(y_pred)
    BCE = K.mean(K.binary_crossentropy(y_true,y_pred))
    dice_loss = 1 - dsc(y_true, y_pred) 
    loss = BCE + dice_loss
    return loss
temp_fpr = []
temp_tpr = []
temp_auc = []
'''def tversky(y_true, y_pred, smooth=1e-6, alpha=0.9):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + 0.5 * false_neg + 0.5 * false_pos + smooth)'''

alpha = [0.5,0.6,0.7,0.8,0.9]
gamma = [1,2,3,1.33]
lr = [0.01,0.001,0.0001,0.000025]
DSC = []

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 0.5)

model.compile(optimizer=AccumOptimizer(Adam(0.0001),2),loss=focal_tversky_loss, metrics = [tversky])
#model.compile(optimizer=Adam(0.00035),loss=focal_tversky_loss, metrics = [tversky])
model.summary()
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=60,validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
model.save("steel_t2")
acc1 = results.history['tversky']
loss1 = results.history['loss']
vacc1 = results.history['val_tversky']
vloss1 = results.history['val_loss']
testing_data_augmnetator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, verbose = 1)
preds_train = (preds_train > 0.5).astype(np.float32)
from sklearn.metrics import roc_curve, roc_auc_score
target_names = ['class 0', 'class 1']
testAcc = K.get_session().run(tversky(true_y, preds_train))
print('DSC: ', testAcc)
df1 = classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names,output_dict=True)
#print(df1)
df1 = pd.DataFrame(df1)
df1.to_csv('t2_Final.csv',index=False)
'''fpr, tpr, _ = roc_curve(true_y.flatten(), preds_train.flatten())
auc = roc_auc_score(true_y.flatten(), preds_train.flatten())
#plt.plot(fpr, tpr,label='α = 0.5, AUC = %.3f' %(auc))
temp_fpr.append(fpr)
temp_tpr.append(tpr)
temp_auc.append(auc)
DSC.append(testAcc)'''
#ImageSave(x_tst, true_y, preds_train, 'Final')
precision, recall, _ = precision_recall_curve(true_y.flatten(), preds_train.flatten())
area = auc(recall,precision)
plt.plot(recall, precision,label='AUC = %.3f' %(area))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('PR Curve')
plt.savefig('steel_t2_PR.png')
plt.clf()
acc1 = results.history['tversky']
loss1 = results.history['loss']
vacc1 = results.history['val_tversky']
vloss1 = results.history['val_loss']

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(acc1, 'r', label='Train')
ax.plot(vacc1, 'b', label='Validation')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_t2_acc_final.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(loss1, 'r', label='Train')
ax.plot(vloss1, 'b', label='Validation')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_t2_loss_final.png')
plt.clf()
'''def tversky(y_true, y_pred, smooth=1e-6, alpha=0.9):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + 0.6 * false_neg + 0.4 * false_pos + smooth)'''
'''def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 0.5)
model = sm.Unet('resnet34', classes = 1, activation='sigmoid', encoder_weights='imagenet')
model.compile(optimizer=AccumOptimizer(Adam(0.0001),2),loss=focal_tversky_loss, metrics = [tversky])
#model.compile(optimizer=Adam(0.00035),loss=focal_tversky_loss, metrics = [tversky])
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=60,validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
#model.save("steel_t2")
acc2 = results.history['tversky']
loss2 = results.history['loss']
vacc2 = results.history['val_tversky']
vloss2 = results.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, verbose = 1)
preds_train = (preds_train > 0.5).astype(np.float32)
from sklearn.metrics import roc_curve, roc_auc_score
target_names = ['class 0', 'class 1']
testAcc = K.get_session().run(tversky(true_y, preds_train))
print('DSC: ', testAcc)
df2 = classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names,output_dict=True)
df2 = pd.DataFrame(df2)
fpr, tpr, _ = roc_curve(true_y.flatten(), preds_train.flatten())
auc = roc_auc_score(true_y.flatten(), preds_train.flatten())
#plt.plot(fpr, tpr,label='γ = 2, AUC = %.3f' %(auc))
temp_fpr.append(fpr)
temp_tpr.append(tpr)
temp_auc.append(auc)
DSC.append(testAcc)
ImageSave(x_tst, true_y, preds_train, 'γ = 2')
def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 1/3)

model = sm.Unet('resnet34', classes = 1, activation='sigmoid', encoder_weights='imagenet')
model.compile(optimizer=AccumOptimizer(Adam(0.0001),2),loss=focal_tversky_loss, metrics = [tversky])
#model.compile(optimizer=Adam(0.00035),loss=focal_tversky_loss, metrics = [tversky])
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=60,validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
#model.save("steel_t2")
acc3 = results.history['tversky']
loss3 = results.history['loss']
vacc3 = results.history['val_tversky']
vloss3 = results.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, verbose = 1)
preds_train = (preds_train > 0.5).astype(np.float32)
from sklearn.metrics import roc_curve, roc_auc_score
target_names = ['class 0', 'class 1']
testAcc = K.get_session().run(tversky(true_y, preds_train))
print('DSC: ', testAcc)
df3 = classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names,output_dict=True)
df3 = pd.DataFrame(df3)
print(classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names))
fpr, tpr, _ = roc_curve(true_y.flatten(), preds_train.flatten())
auc = roc_auc_score(true_y.flatten(), preds_train.flatten())
#plt.plot(fpr, tpr,label='γ = 3, AUC = %.3f' %(auc))
temp_fpr.append(fpr)
temp_tpr.append(tpr)
temp_auc.append(auc)
DSC.append(testAcc)
ImageSave(x_tst, true_y, preds_train, 'γ = 3')

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 0.75)
model = sm.Unet('resnet34', classes = 1, activation='sigmoid', encoder_weights='imagenet')
model.compile(optimizer=AccumOptimizer(Adam(0.0001),2),loss=focal_tversky_loss, metrics = [tversky])
#model.compile(optimizer=Adam(0.00035),loss=focal_tversky_loss, metrics = [tversky])
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=60,validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
#model.save("steel_t2")
acc4 = results.history['tversky']
loss4 = results.history['loss']
vacc4 = results.history['val_tversky']
vloss4 = results.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, verbose = 1)
preds_train = (preds_train > 0.5).astype(np.float32)
from sklearn.metrics import roc_curve, roc_auc_score
target_names = ['class 0', 'class 1']
testAcc = K.get_session().run(tversky(true_y, preds_train))
print('DSC: ', testAcc)
df4 = classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names,output_dict=True)
df4 = pd.DataFrame(df4)
fpr, tpr, _ = roc_curve(true_y.flatten(), preds_train.flatten())
auc = roc_auc_score(true_y.flatten(), preds_train.flatten())
#plt.plot(fpr, tpr,label='γ = 1.33, AUC = %.3f' %(auc))
temp_fpr.append(fpr)
temp_tpr.append(tpr)
temp_auc.append(auc)
DSC.append(testAcc)
ImageSave(x_tst, true_y, preds_train, 'γ = 1.33')'''

'''def tversky(y_true, y_pred, smooth=1e-6, alpha=0.9):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + 0.9 * false_neg + 0.1 * false_pos + smooth)'''

'''def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 0.75)'''
'''model = sm.Unet('resnet34', classes = 1, activation='sigmoid', encoder_weights='imagenet')
model.compile(optimizer=AccumOptimizer(Adam(0.0001),2),loss=tversky_loss, metrics = [tversky])
#model.compile(optimizer=Adam(0.00035),loss=focal_tversky_loss, metrics = [tversky])
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=60,validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
#model.save("steel_t2")
acc5 = results.history['tversky']
loss5 = results.history['loss']
vacc5 = results.history['val_tversky']
vloss5 = results.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict(x_tst, verbose = 1)
preds_train = (preds_train > 0.5).astype(np.float32)
from sklearn.metrics import roc_curve, roc_auc_score
target_names = ['class 0', 'class 1']
testAcc = K.get_session().run(tversky(true_y, preds_train))
print('DSC: ', testAcc)
df5 = classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names,output_dict=True)
df5 = pd.DataFrame(df5)
print(classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names))
fpr, tpr, _ = roc_curve(true_y.flatten(), preds_train.flatten())
auc = roc_auc_score(true_y.flatten(), preds_train.flatten())
#plt.plot(fpr, tpr,label='α = 0.5, AUC = %.3f' %(auc))
temp_fpr.append(fpr)
temp_tpr.append(tpr)
temp_auc.append(auc)
DSC.append(testAcc)
ImageSave(x_tst, true_y, preds_train, 'α = 0.9')'''

i = 0
'''for beta in [0.5, 0.6, 0.7, 0.8, 0.9]:
    plt.plot(temp_fpr[i], temp_tpr[i],label='α = %.1f, AUC = %.3f' %(beta, temp_auc[i]))
    i = i + 1'''
'''for beta in [1,2,3,1.33]:
    plt.plot(temp_fpr[i], temp_tpr[i],label='γ = %.2f, AUC = %.3f' %(beta, temp_auc[i]))
    i = i + 1


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.savefig('steel_t2_ROC_gamma.png')
plt.clf()'''

'''for beta in [0.01,0.001,0.0001,0.000025]:
   plt.plot(temp_fpr[i], temp_tpr[i],label='lr = %.2f, AUC = %.3f' %(beta, temp_auc[i]))
    i = i + 1

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.savefig('steel_t2_ROC_lr.png')
plt.clf()'''

'''fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(acc1, 'r', label='γ = 1')
ax.plot(acc2, 'g', label='γ = 2')
ax.plot(acc3, 'y', label='γ = 3')
ax.plot(acc4, 'c', label='γ = 1.33')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_t2_acc_gamma.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(loss1, 'r', label='γ = 1')
ax.plot(loss2, 'g', label='γ = 2')
ax.plot(loss3, 'y', label='γ = 3')
ax.plot(loss4, 'c', label='γ = 1.33')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_t2_loss_gamma.png')
plt.clf()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vacc1, 'r', label='γ = 1')
ax.plot(vacc2, 'g', label='γ = 2')
ax.plot(vacc3, 'y', label='γ = 3')
ax.plot(vacc4, 'c', label='γ = 1.33')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_t2_vacc_gamma.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vloss1, 'r', label='γ = 1')
ax.plot(vloss2, 'g', label='γ = 2')
ax.plot(vloss3, 'y', label='γ = 3')
ax.plot(vloss4, 'c', label='γ = 1.33')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_t2_vloss_gamma.png')
plt.clf()

DF = pd.concat([df1,df2,df3,df4],axis=0)
filename = 'steel_t2_report_gamma.csv'
DF.to_csv(filename,index=False)

submission = pd.DataFrame({'Gamma': gamma, 'DSC': DSC})
filename = 'steel_t2_DSC_gamma.csv'
submission.to_csv(filename,index=False)'''
'''def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 0.75)

model.compile(optimizer=AccumOptimizer(Adam(0.001),4),loss=focal_tversky_loss, metrics = [tversky])
#model.compile(optimizer=Adam(0.00035),loss=focal_tversky_loss, metrics = [tversky])
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=120,validation_data=val_training_data_generator, validation_steps=len(x_val)//8)
#model.save("steel_t2")
acc1 = results.history['tversky']
loss1 = results.history['loss']
testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
preds_train = model.predict_generator(testing_data_augmentator, verbose = 1)

target_names = ['class 0', 'class 1']
print(classification_report(true_y.flatten(), preds_train.flatten(), target_names=target_names))'''
#preds_train = (preds_train > 0.5).astype(np.float32)

'''testAcc = K.get_session().run(tversky(true_y, preds_train))
print('DSC: ', testAcc)
recall, precision, f1score = K.get_session().run(my_metrics(true_y,preds_train))
print("Precision : ", precision)
print("f1score : ", f1score)
print("Recall : ", recall)


print('LR: 0.000025 / Batch Size = 32')
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    preds = (i < preds_train).astype(np.float32)
    print(i)
    testAcc = K.get_session().run(tversky(true_y, preds))
    print('DSC: ', testAcc)
    recall, precision, f1score = K.get_session().run(my_metrics(true_y,preds))
    print("Precision : ", precision)
    print("f1score : ", f1score)
    print("Recall : ", recall)'''

'''fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(results.history['loss'], 'r', label='train')
ax.plot(results.history['val_loss'], 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
fig.savefig('steel_t2_loss_whole.png')
plt.clf()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(results.history['tversky'], 'r', label='train')
ax.plot(results.history['val_tversky'], 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
fig.savefig('steel_t2_acc_whole.png')
plt.clf()'''

'''for i in range(3):
    plt.subplot(3, 3, 1 + i)
    plt.axis('off')
    plt.imshow(x_tst[i].astype(np.int32))
    #plt.imshow(x_tst[i],cmap='gray')
    # plot generted target image
for i in range(3): 
    plt.subplot(3, 3, 1 + 3 + i)
    plt.axis('off')
    plt.imshow(true_y[i],cmap='gray')
    # plot real target image
for i in range(3):
    plt.subplot(3, 3, 1 + 3*2 + i)
    plt.axis('off')
    plt.imshow(preds_train[i],cmap='gray')
filename1 = 't2_whole1(FL).png'
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
filename1 = 't2_whole2(FL).png'
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
filename1 = 't2_whole3(FL).png'
plt.savefig(filename1)
plt.close()'''

#df = pd.DataFrame(t2_defect,columns=['per'])
#print(df.head(5))
#df['per'].value_counts().plot(kind='bin')
#plt.show()
#plt.hist(df['per'].value_counts())
#plt.show()


