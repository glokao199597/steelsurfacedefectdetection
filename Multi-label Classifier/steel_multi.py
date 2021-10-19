

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
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, History, CSVLogger
from PIL import Image
from PIL import ImageFilter
from skimage.io import imshow
from skimage.util import random_noise
from sklearn.utils import class_weight
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, precision_recall_curve,recall_score,average_precision_score,auc
import segmentation_models as sm
import random
import albumentations
from ImageDataAugmentor.image_data_augmentor import *
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sn
from sklearn.metrics import roc_curve, roc_auc_score
from skimage.feature import hog
keras.backend.set_image_data_format('channels_last')
from cosine_annealing import CosineAnnealingScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import , Dense, BatchNormalization, Dropout


seed = 42
T_max=50
eta_max=2e-3
eta_min=1e-5
flat=30
epochs = 30
learning_rate = 0.0001 
decay_rate = 0.0001
momentum = 0.8

def lr_scheduler(epoch):
    if epoch>flat:
        temp=(epoch-flat)%T_max
        temp2=(epoch-flat)//T_max
        lr = eta_min + 0.5*(eta_max - eta_min)*(1+math.cos(temp/T_max*math.pi))*(0.8**temp2)
    else:
        lr=eta_max
    return lr

lr_rate = LearningRateScheduler(lr_scheduler)
csvlogger = CSVLogger('steel-all.log')
callbacks_list =[csvlogger,lr_rate]

def create_classification_report(y_true, y_preds, opt, defect):
    preds = (y_preds > 0.2).astype(np.int)
    df1 = classification_report(y_true, preds, labels=[0, 1],output_dict=True)
    df1 = pd.DataFrame(df1)
    preds = (y_preds > 0.3).astype(np.int)
    df2 = classification_report(y_true, preds, labels=[0, 1],output_dict=True)
    df2 = pd.DataFrame(df2)
    preds = (y_preds > 0.4).astype(np.int)
    df3 = classification_report(y_true, preds, labels=[0, 1],output_dict=True)
    df3 = pd.DataFrame(df3)
    preds = (y_preds > 0.5).astype(np.int)
    df4 = classification_report(y_true, preds, labels=[0, 1],output_dict=True)
    df4 = pd.DataFrame(df4)
    preds = (y_preds > 0.6).astype(np.int)
    df5 = classification_report(y_true, preds, labels=[0, 1],output_dict=True)
    df5 = pd.DataFrame(df5)
    preds = (y_preds > 0.7).astype(np.int)
    df6 = classification_report(y_true, preds, labels=[0, 1],output_dict=True)
    df6 = pd.DataFrame(df6)
    preds = (y_preds > 0.8).astype(np.int)
    df7 = classification_report(y_true, preds, labels=[0, 1],output_dict=True)
    df7 = pd.DataFrame(df7)
    preds = (y_preds > 0.9).astype(np.int)
    df8 = classification_report(y_true, preds, labels=[0, 1],output_dict=True)
    df8 = pd.DataFrame(df8)
    DF = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8],axis=0)
    filename = ('steel_multi_report_' + opt + '_%d.csv' % defect)
    DF.to_csv(filename,index=False)


def my_metrics(y_true, y_pred, model):
    temp_recall = []
    temp_precision = []
    temp_auc = []
    i = 0
    for j in range(0,4):
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            preds = (y_pred[:,j]> threshold).astype(np.int)
            cm=confusion_matrix(y_true[:,j], preds)
            plt.subplot(2, 4, 1 + i)
            sn.heatmap(cm, annot=True, fmt='d', cmap = 'Blues', xticklabels=['Non-Defect','Defect'], yticklabels=['Non-Defect','Defect'])# font size
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Threshold = %.1f' % (threshold), fontsize=10)
            precision, recall, _ = precision_recall_curve(y_true[:,j], preds)
            area = auc(recall,precision)
            temp_recall.append(recall)
            temp_precision.append(precision)
            temp_auc.append(area)
            i = i + 1
        plt.tight_layout()
        plt.savefig('steel_multi_' + model +  '_%d.png' % (j+1))
        plt.close()
        create_classification_report(y_true[:,j], y_pred[:,j], model,j+1)
        i = 0
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            plt.plot(temp_recall[i], temp_precision[i], lw=2, label='Thres = %.1f, AUC = %.3f' %(threshold,temp_auc[i]))
            i = i + 1
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.title('PR Curve')
        plt.savefig('steel_multi_PR' + model + '_%d.png' % (j+1))
        plt.clf()
        temp_recall = []
        temp_precision = []
        temp_auc = []
        i = 0 




def rle2mask(mask_rle, i, shape=(1600,256)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype =np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = i
    img = img.reshape(shape).T
    img = np.reshape(img,(256,1600,-1))
    return img

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



# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def exp_decay(epoch):
    lrate = learning_rate*np.exp(-decay_rate*epoch)
    print(lrate)
    return lrate

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

#train_data['powerlabel'] = train_data.apply(lambda x : 8*x['H4']+4*x['H3']+2*x['H2']+1*x['H1'],axis=1)
print((train_data['H1']==1).sum())
print((train_data['H2']==1).sum())
print((train_data['H3']==1).sum())
print((train_data['H4']==1).sum())

train_data = train_data[['ImageId','D1','D2','D3','D4']][train_data['Defect']==1]
print(train_data.head(10))
X_train = np.zeros((len(train_data),256,1600,3),dtype=np.uint8)
Y_train = np.zeros((len(train_data),256,1600,1),dtype=np.uint8)
#Y_train = np.zeros((len(train_data),256,1600),dtype=np.uint8)
Y_train_multi = []
idx = 0
num = 0
for idx, i in enumerate(train_data.values,0):
    H1 = 0
    H2 = 0
    H3 = 0
    H4 = 0
    num = 0 
    temp_Y = np.zeros((256,1600,1),dtype=np.uint8)
    mask = np.zeros((256,1600,1),dtype=np.uint8)
    if(i[1]!=0):
        mask = (rle2mask(i[1],1))
        num = num + 1
    if(i[2]!=0):
        temp_Y = (rle2mask(i[2],2))
        mask = np.maximum(mask,temp_Y)
        num = num+1        
    if(i[3]!=0):
        temp_Y = (rle2mask(i[3],3))
        mask = np.maximum(mask,temp_Y) 
        num = num + 1
    if(i[4]!=0):
        temp_Y = (rle2mask(i[4],4))
        mask = np.maximum(mask,temp_Y) 
        num = num + 1
    Y_train[idx] = np.reshape(mask,(256,1600,-1))

    img = cv2.imread(train_path + i[0],0)
    bilateral = cv2.bilateralFilter(img, 3, 150, 300)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    temp_X = clahe.apply(img)
    temp_X = cv2.cvtColor(temp_X, cv2.COLOR_GRAY2RGB)
    X_train[idx] = temp_X
rmv_idx = []
new_X_train = np.zeros((len(X_train)*2,256,800,3),dtype=np.uint8)
new_Y_train = np.zeros((len(X_train)*2,256,800,1),dtype=np.uint8)
i = 0
print(len(new_X_train))
print(len(new_Y_train))
for a , b in zip(X_train,Y_train):
    H1 = 0
    H2 = 0
    H3 = 0
    H4 = 0
    num = 0
    new_X_train[i] = a[:,:800,:]
    new_Y_train[i] = b[:,:800,:]
    if(np.all(new_Y_train[i]==0)):
        rmv_idx.append(i)
    if(np.any(new_Y_train[i]==1)):
        H1 = 1
    if(np.any(new_Y_train[i]==2)):
        H2 = 1
    if(np.any(new_Y_train[i]==3)):
        H3 = 1
    if(np.any(new_Y_train[i]==4)):
        H4 = 1
    i = i + 1
    Y_train_multi.append([H1, H2 ,H3, H4])
    H1 = 0
    H2 = 0
    H3 = 0
    H4 = 0
    new_X_train[i] = a[:,800:1600,:]
    new_Y_train[i] = b[:,800:1600,:]
    if(np.all(new_Y_train[i]==0)):
        rmv_idx.append(i)
    if(np.any(new_Y_train[i]==1)):
        H1 = 1
    if(np.any(new_Y_train[i]==2)):
        H2 = 1
    if(np.any(new_Y_train[i]==3)):
        H3 = 1
    if(np.any(new_Y_train[i]==4)):
        H4 = 1
    i = i + 1
    Y_train_multi.append([H1, H2 ,H3, H4])

new_X_train = np.delete(new_X_train,rmv_idx,0)
Y_train_multi = np.delete(Y_train_multi,rmv_idx,0)
print((Y_train_multi[:,0]==1).sum())
print((Y_train_multi[:,1]==1).sum())
print((Y_train_multi[:,2]==1).sum())
print((Y_train_multi[:,3]==1).sum())

'''X_norm = new_X_train/255.0
new_X_train = new_X_train - np.array(np.mean(X_norm,axis=(0,1,2)))
new_X_train = new_X_train/np.array(np.mean(X_norm,axis=(0,1,2)))
new_X_train = new_X_train.astype(np.float32)'''
print(new_X_train.shape)
print(Y_train_multi.shape)

x_tra, x_tst, y_tra, y_tst = train_test_split(new_X_train, Y_train_multi, test_size = 0.1, shuffle = True, random_state  = 123)
x_tra, x_val, y_tra, y_val = train_test_split(x_tra, y_tra, test_size = 0.2, shuffle = True, random_state  = 123)



Aug = albumentations.Compose([
                 albumentations.OneOf([        
                 albumentations.HorizontalFlip(p=0.5), 
                 #albumentations.GridDistortion(num_steps=5, distort_limit=(0,0.03),p=0.5),
                 albumentations.VerticalFlip(p=0.5),
                 albumentations.RandomResizedCrop(200, 200, scale=(0.4, 0.8), p=0.5)],p=1)])
                 #albumentations.OpticalDistortion(distort_limit=0.03, shift_limit=0.03,p=0.5)

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

outputs = Dense(4, activation='sigmoid')(x)

model_in = Model(base_model.input,outputs)



sgd = SGD(lr=0.1, momentum=0.9)
callbacks = [
    CosineAnnealingScheduler(T_max=10, eta_max=1e-3, eta_min=1e-5)
]

model_in.compile(optimizer=sgd,
loss='binary_crossentropy',
metrics=['accuracy'])

training_data_augmentator = training_datagen.flow(x_tra, y_tra, batch_size=8, shuffle=False)
validation_data_augmentator = validation_datagen.flow(x_val, y_val, batch_size=8, shuffle=False)
results_x = model_in.fit_generator(training_data_augmentator, steps_per_epoch=len(x_tra)//8, epochs=30, callbacks =callbacks,  validation_data = validation_data_augmentator)
#model_in.save("steel_binary")
acc1 = results_x.history['accuracy']
loss1 = results_x.history['loss']
vacc1 = results_x.history['val_accuracy']
vloss1 = results_x.history['val_loss']
testing_data_augmentator = testing_datagen.flow(x_tst, y_tst, batch_size=8, shuffle=False)
preds = model_in.predict_generator(testing_datagen.flow(x_tst, batch_size=8, shuffle=False), verbose = 1)
#preds = preds.astype(np.uint8)
true_y = testing_data_augmentator.y


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

my_metrics(true_y,preds,'in_exp')

model_in = Model(base_model.input,outputs)

model_in.compile(optimizer=Adam(0.001),
loss='binary_crossentropy',
metrics=['accuracy'])



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

my_metrics(true_y,preds,'in_adam')


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(acc1, 'r', label='Cosine Delay')
ax.plot(acc2, 'g', label='Exp. Decay')
ax.plot(acc3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_multi_acc_in.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(loss1, 'r', label='Cosine Delay')
ax.plot(loss2, 'g', label='Exp. Decay')
ax.plot(loss3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_multi_loss_in.png')
plt.clf()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vacc1, 'r', label='Cosine Delay')
ax.plot(vacc2, 'g', label='Exp. Decay')
ax.plot(vacc3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_multi_vacc_in.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vloss1, 'r', label='Cosine Delay')
ax.plot(vloss2, 'g', label='Exp. Decay')
ax.plot(vloss3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_multi_vloss_in.png')
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

outputs = Dense(4, activation='sigmoid')(x)

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


training_data_augmentator = training_datagen.flow(x_tra, y_tra, batch_size=8, shuffle=False)
validation_data_augmentator = validation_datagen.flow(x_val, y_val, batch_size=8, shuffle=False)
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
my_metrics(true_y,preds,'x_exp')

model_x = Model(base_model.input,outputs)

model_x.compile(optimizer=Adam(0.001),
loss='binary_crossentropy',
metrics=['accuracy'])


training_data_augmentator = training_datagen.flow(x_tra, y_tra, batch_size=8, shuffle=False)
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

my_metrics(true_y,preds,'x_adam')


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(acc1, 'r', label='Cosine Delay')
ax.plot(acc2, 'g', label='Exp. Decay')
ax.plot(acc3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_multi_acc_x.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(loss1, 'r', label='Cosine Delay')
ax.plot(loss2, 'g', label='Exp. Decay')
ax.plot(loss3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_multi_loss_x.png')
plt.clf()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vacc1, 'r', label='Cosine Delay')
ax.plot(vacc2, 'g', label='Exp. Decay')
ax.plot(vacc3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Dice Score', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_multi_vacc_x.png')
plt.clf()


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(vloss1, 'r', label='Cosine Delay')
ax.plot(vloss2, 'g', label='Exp. Decay')
ax.plot(vloss3, 'y', label='Adam')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.grid()
fig.savefig('steel_multi_vloss_x.png')
plt.clf()






