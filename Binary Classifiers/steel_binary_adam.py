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
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
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
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, History
from PIL import Image
from PIL import ImageFilter
from skimage.io import imshow
from skimage.util import random_noise
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import albumentations
from ImageDataAugmentor.image_data_augmentor import *
import json
import seaborn as sn
from sklearn.metrics import roc_curve, roc_auc_score
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from cosine_annealing import CosineAnnealingScheduler
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import , Dense, BatchNormalization, Dropout
seed = 42

epochs = 30
learning_rate = 0.0001 
decay_rate = 0.0001
momentum = 0.8


def my_metrics(y_true, y_pred, model):
    temp_tpr = []
    temp_auc = []
    temp_fpr = []
    i = 0 
    for threshold in [0.5, 0.55, 0.6, 0.65,  0.7, 0.75, 0.8,0.85, 0.9]:
        preds = (y_pred > threshold).astype(np.int)
        cm=confusion_matrix(y_true, preds)
        
        plt.subplot(3, 3, 1 + i)
        plt.axis('off')
        sn.heatmap(cm, annot=True, fmt='d', cmap = 'Blues', xticklabels=['Non-Defect','Defect'], yticklabels=['Non-Defect','Defect'])# font size
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Threshold = %.2f' % (i))
        fpr, tpr, _ = roc_curve(y_true, preds)
        auc = roc_auc_score(y_true, preds)
        temp_fpr.append(fpr)
        temp_tpr.append(tpr)
        temp_auc.append(auc)
        i = i + 1

    plt.savefig('steel_binary_' + model + '.png')
    plt.close()
    i = 0 
    for beta in [0.5, 0.55, 0.6, 0.65,  0.7, 0.75, 0.8, 0.85, 0.9]:
        plt.plot(temp_fpr[i], temp_tpr[i],label='Thre = %.2f, AUC = %.3f' %(beta, temp_auc[i]))
        i = i + 1

    plt.xlabel('False Positive Rate')   
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curve')
    plt.savefig('steel_binary_ROC' + model + '.png')
    plt.clf()


def rle2mask(mask_rle, i, shape=(1600,256)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype =np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = i
    return img.reshape(shape).T

def create_classification_report(y_true, y_preds, opt):
    preds = (y_preds > 0.5).astype(np.int)
    df1 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df1 = pd.DataFrame(df1)
    preds = (y_preds > 0.55).astype(np.int)
    df2 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df2 = pd.DataFrame(df2)
    preds = (y_preds > 0.6).astype(np.int)
    df3 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df3 = pd.DataFrame(df3)
    preds = (y_preds > 0.65).astype(np.int)
    df4 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df4 = pd.DataFrame(df4)
    preds = (y_preds > 0.7).astype(np.int)
    df5 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df5 = pd.DataFrame(df5)
    preds = (y_preds > 0.75).astype(np.int)
    df6 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df6 = pd.DataFrame(df6)
    preds = (y_preds > 0.8).astype(np.int)
    df7 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df7 = pd.DataFrame(df7)
    preds = (y_preds > 0.85).astype(np.int)
    df8 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df8 = pd.DataFrame(df8)
    preds = (y_preds > 0.9).astype(np.int)
    df9 = classification_report(y_true, preds, target_names=['Class 0', 'Class 1'],output_dict=True)
    df9 = pd.DataFrame(df9)
    DF = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9],axis=0)
    filename = 'steel_binary_report_' + opt + '.csv'
    DF.to_csv(filename,index=False)



train_path = 'severstal-steel-defect-detection/train_images/'
test_path = 'severstal-steel-defect-detection/test_images/'

train_images = next(os.walk(train_path))[2]
train_images.sort()
test_images = next(os.walk(test_path))[2]

X_train_bin = np.zeros((len(train_images),256,1600,3),dtype=np.uint8)

train_path = 'severstal-steel-defect-detection/train_images/'
for n, image in enumerate(train_images, 0):
    temp_img = cv2.imread(train_path + image)
    X_train_bin[n] = temp_img


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
print(train_data.isnull().sum())
#train_data['no_of_defect'].value_counts().sort_index(ascending=True).plot('bar')
Y_train = train_data['Defect']
print(Y_train)
SEED = 123
train_data = train_data['ImageId']

X_train = np.zeros((len(train_data),256,800,3),dtype=np.uint8)


for idx, i in enumerate(train_data.values,0):
    img = cv2.imread(train_path + i ,0)
    bilateral = cv2.bilateralFilter(img, 3, 150, 300)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    temp_X = clahe.apply(img)
    temp_X = cv2.cvtColor(temp_X, cv2.COLOR_GRAY2RGB)
    X_train[idx] = cv2.resize(temp_X,(800,256),interpolation = cv2.INTER_CUBIC)


Aug = albumentations.Compose([
                 albumentations.OneOf([        
                 albumentations.HorizontalFlip(p=0.5), 
                 albumentations.GridDistortion(num_steps=5, distort_limit=(0,0.03),p=0.5),
                 albumentations.VerticalFlip(p=0.5),
                 albumentations.RandomResizedCrop(200, 200, scale=(0.4, 0.8), p=0.5),
                 albumentations.OpticalDistortion(distort_limit=0.03, shift_limit=0.03,p=0.5)],p=1)])

training_datagen = ImageDataAugmentor(augment = Aug, augment_seed = 123)
validation_datagen = ImageDataAugmentor()
testing_datagen = ImageDataAugmentor()

training_datagen = ImageDataAugmentor()

validation_datagen = ImageDataAugmentor()
testing_datagen = ImageDataAugmentor()

base_model = InceptionV3(input_shape=(None, None, 3), weights='imagenet', include_top=False)

# Top Model Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
          
x = Dense(1024,activation='relu')(x)


x = Dense(512,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)

outputs = Dense(1, activation='sigmoid')(x)

model_in = Model(base_model.input,outputs)


x_tra, x_tst, y_tra, y_tst = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 42, stratify = Y_train)

x_tra, x_val, y_tra, y_val = train_test_split(x_tra, y_tra, test_size = 0.2, random_state = 42, stratify = y_tra)

model_in = Model(base_model.input,outputs)

model_in.compile(optimizer=Adam(0.0001),
loss='binary_crossentropy',
metrics=['accuracy'])


training_data_augmentator = training_datagen.flow(x_tra, y_tra, batch_size=8, shuffle=False)
validation_data_augmentator = validation_datagen.flow(x_val, y_val, batch_size=8, shuffle=False)
results_x = model_in.fit_generator(training_data_augmentator, steps_per_epoch=len(x_tra)//8, epochs=30,  validation_data = validation_data_augmentator)
model_in.save("steel_binary")
acc3 = results_x.history['accuracy']
loss3 = results_x.history['loss']
vacc3 = results_x.history['val_accuracy']
vloss3 = results_x.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, y_tst, batch_size=8, shuffle=False)
preds = model_in.predict_generator(testing_datagen.flow(x_tst, batch_size=8, shuffle=False), verbose = 1)
#preds = preds.astype(np.uint8)
true_y = testing_data_augmentator.y
create_classification_report(true_y,preds,'in_adam')
my_metrics(true_y,preds,'in_adam')

sgd = SGD(lr=0.1, momentum=0.9)
callbacks = [
    CosineAnnealingScheduler(T_max=10, eta_max=1e-3, eta_min=1e-5)
]

model_in.compile(optimizer=sgd,
loss='binary_crossentropy',
metrics=['accuracy'])

results_x = model_in.fit_generator(training_data_augmentator, steps_per_epoch=len(x_tra)//8, epochs=30, callbacks =callbacks,  validation_data = validation_data_augmentator)
model_in.save("steel_binary")
acc1 = results_x.history['accuracy']
loss1 = results_x.history['loss']
vacc1 = results_x.history['val_accuracy']
vloss1 = results_x.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, y_tst, batch_size=8, shuffle=False)
preds = model_in.predict_generator(testing_datagen.flow(x_tst, batch_size=8, shuffle=False), verbose = 1)
#preds = preds.astype(np.uint8)
true_y = testing_data_augmentator.y

create_classification_report(true_y,preds,'in_cas')
my_metrics(true_y,preds,'in_cas')

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

def exp_decay(epoch):
    lrate = learning_rate*np.exp(-decay_rate*epoch)
    print(lrate)
    return lrate

lrate = LearningRateScheduler(exp_decay)
callbacks = [lrate]

model_in = Model(base_model.input,outputs)

model_in.compile(optimizer=sgd,
loss='binary_crossentropy',
metrics=['accuracy'])


results_x = model_in.fit_generator(training_data_augmentator, steps_per_epoch=len(x_tra)//8, epochs=30, callbacks =callbacks,  validation_data = validation_data_augmentator)
model_in.save("steel_binary")
acc2 = results_x.history['accuracy']
loss2 = results_x.history['loss']
vacc2 = results_x.history['val_accuracy']
vloss2 = results_x.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, y_tst, batch_size=8, shuffle=False)
preds = model_in.predict_generator(testing_datagen.flow(x_tst, batch_size=8, shuffle=False), verbose = 1)
#preds = preds.astype(np.uint8)
true_y = testing_data_augmentator.y
create_classification_report(true_y,preds,'in_exp')
my_metrics(true_y,preds,'in_exp')



fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(acc1, 'r', label='Cosine Delay')
ax.plot(acc2, 'g', label='Exp. Decay')
ax.plot(acc3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_binary_acc_in.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(loss1, 'r', label='Cosine Delay')
ax.plot(loss2, 'g', label='Exp. Decay')
ax.plot(loss3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_binary_loss_in.png')
plt.clf()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vacc1, 'r', label='Cosine Delay')
ax.plot(vacc2, 'g', label='Exp. Decay')
ax.plot(vacc3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_binary_vacc_in.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vloss1, 'r', label='Cosine Delay')
ax.plot(vloss2, 'g', label='Exp. Decay')
ax.plot(vloss3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_binary_vloss_in.png')
plt.clf()



base_model = Xception(input_shape=(None, None, 3), weights='imagenet', include_top=False)

# Top Model Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
          
x = Dense(1024,activation='relu')(x)


x = Dense(512,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)

outputs = Dense(1, activation='sigmoid')(x)

model_x = Model(base_model.input,outputs)


sgd = SGD(lr=0.1, momentum=0.9)
callbacks = [
    CosineAnnealingScheduler(T_max=10, eta_max=1e-3, eta_min=1e-5)
]

model_x.compile(optimizer=sgd,
loss='binary_crossentropy',
metrics=['accuracy'])


results_x = model_x.fit_generator(training_data_augmentator, steps_per_epoch=len(x_tra)//8, epochs=30, callbacks =callbacks,  validation_data = validation_data_augmentator)
model_x.save("steel_binary")
acc1 = results_x.history['accuracy']
loss1 = results_x.history['loss']
vacc1 = results_x.history['val_accuracy']
vloss1 = results_x.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, y_tst, batch_size=8, shuffle=False)
preds = model_x.predict_generator(testing_datagen.flow(x_tst, batch_size=8, shuffle=False), verbose = 1)
#preds = preds.astype(np.uint8)
true_y = testing_data_augmentator.y

create_classification_report(true_y,preds,'x_cas')
my_metrics(true_y,preds,'x_cas')


sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

def exp_decay(epoch):
    lrate = learning_rate*np.exp(-decay_rate*epoch)
    print(lrate)
    return lrate

lrate = LearningRateScheduler(exp_decay)
callbacks = [lrate]

model_x = Model(base_model.input,outputs)

model_x.compile(optimizer=sgd,
loss='binary_crossentropy',
metrics=['accuracy'])



results_x = model_x.fit_generator(training_data_augmentator, steps_per_epoch=len(x_tra)//8, epochs=30, callbacks =callbacks,  validation_data = validation_data_augmentator)
model_x.save("steel_binary")
acc2 = results_x.history['accuracy']
loss2 = results_x.history['loss']
vacc2 = results_x.history['val_accuracy']
vloss2 = results_x.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, y_tst, batch_size=8, shuffle=False)
preds = model_x.predict_generator(testing_datagen.flow(x_tst, batch_size=8, shuffle=False), verbose = 1)
#preds = preds.astype(np.uint8)
true_y = testing_data_augmentator.y
create_classification_report(true_y,preds,'x_exp')
my_metrics(true_y,preds,'x_exp')

model_x = Model(base_model.input,outputs)

model_x.compile(optimizer=Adam(0.0001),
loss='binary_crossentropy',
metrics=['accuracy'])


results_x = model_x.fit_generator(training_data_augmentator, steps_per_epoch=len(x_tra)//8, epochs=30,  validation_data = validation_data_augmentator)
model_x.save("steel_binary")
acc3 = results_x.history['accuracy']
loss3 = results_x.history['loss']
vacc3 = results_x.history['val_accuracy']
vloss3 = results_x.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, y_tst, batch_size=8, shuffle=False)
preds = model_x.predict_generator(testing_datagen.flow(x_tst, batch_size=8, shuffle=False), verbose = 1)
#preds = preds.astype(np.uint8)
true_y = testing_data_augmentator.y
create_classification_report(true_y,preds,'x_adam')
my_metrics(true_y,preds,'x_adam')


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(acc1, 'r', label='Cosine Delay')
ax.plot(acc2, 'g', label='Exp. Decay')
ax.plot(acc3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_binary_acc_x.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(loss1, 'r', label='Cosine Delay')
ax.plot(loss2, 'g', label='Exp. Decay')
ax.plot(loss3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_binary_loss_x.png')
plt.clf()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vacc1, 'r', label='Cosine Delay')
ax.plot(vacc2, 'g', label='Exp. Decay')
ax.plot(vacc3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_binary_vacc_x.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vloss1, 'r', label='Cosine Delay')
ax.plot(vloss2, 'g', label='Exp. Decay')
ax.plot(vloss3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_binary_vloss_x.png')
plt.clf()



'''fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(adam_val_accuracy, 'r', label='Adam')
ax.plot(sgd_val_accuracy, 'b', label='SGD')
ax.plot(rms_val_accuracy, 'g', label='RMSprop')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Val_Acc', fontsize=20)
ax.legend()
fig.savefig('val_accuracy_all.png')
plt.clf()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(adam_val_loss, 'r', label='Adam')
ax.plot(sgd_val_loss, 'b', label='SGD')
ax.plot(rms_val_loss, 'g', label='RMSprop')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Val_Loss', fontsize=20)
ax.legend()
fig.savefig('val_loss_all.png')
plt.clf()'''
