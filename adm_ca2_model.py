#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, shutil
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from sklearn.model_selection import learning_curve,train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from skimage import img_as_ubyte
from skimage.exposure import histogram
from skimage.color import rgb2gray

#Project Configuration
PROJ_WD='C:/Users/darsh/Documents/SEM2/ADM/ADM/Project'
DATA_DIR=PROJ_WD+'/full_train'  #original_dataset_dir
SMP_DATA_DIR=PROJ_WD+'/full_train_sample'  #original_dataset_dir
LBL_MAP_FILE=PROJ_WD+'/trainLabels.csv'
TL_CNN=PROJ_WD+'/TLCNN' #base_dir 

SAMPL=5000


# In[ ]:


train_dir = os.path.join(TL_CNN,'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
    
validation_dir = os.path.join(TL_CNN,'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

test_dir = os.path.join(TL_CNN,'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
    
train_stage0_dir = os.path.join(train_dir,'stage0')
if not os.path.exists(train_stage0_dir):
    os.mkdir(train_stage0_dir)

train_stage1_dir = os.path.join(train_dir,'stage1')
if not os.path.exists(train_stage1_dir):
    os.mkdir(train_stage1_dir)
    
train_stage2_dir = os.path.join(train_dir,'stage2')
if not os.path.exists(train_stage2_dir):
    os.mkdir(train_stage2_dir)
    
train_stage3_dir = os.path.join(train_dir,'stage3')
if not os.path.exists(train_stage3_dir):
    os.mkdir(train_stage3_dir)
    
train_stage4_dir = os.path.join(train_dir,'stage4')
if not os.path.exists(train_stage4_dir):
    os.mkdir(train_stage4_dir)
    
val_stage0_dir = os.path.join(validation_dir,'stage0')
if not os.path.exists(val_stage0_dir):
    os.mkdir(val_stage0_dir)

val_stage1_dir = os.path.join(validation_dir,'stage1')
if not os.path.exists(val_stage1_dir):
    os.mkdir(val_stage1_dir)
    
val_stage2_dir = os.path.join(validation_dir,'stage2')
if not os.path.exists(val_stage2_dir):
    os.mkdir(val_stage2_dir)
    
val_stage3_dir = os.path.join(validation_dir,'stage3')
if not os.path.exists(val_stage3_dir):
    os.mkdir(val_stage3_dir)
    
val_stage4_dir = os.path.join(validation_dir,'stage4')
if not os.path.exists(val_stage4_dir):
    os.mkdir(val_stage4_dir)
    
test_stage0_dir = os.path.join(test_dir,'stage0')
if not os.path.exists(test_stage0_dir):
    os.mkdir(test_stage0_dir)

test_stage1_dir = os.path.join(test_dir,'stage1')
if not os.path.exists(test_stage1_dir):
    os.mkdir(test_stage1_dir)
    
test_stage2_dir = os.path.join(test_dir,'stage2')
if not os.path.exists(test_stage2_dir):
    os.mkdir(test_stage2_dir)
    
test_stage3_dir = os.path.join(test_dir,'stage3')
if not os.path.exists(test_stage3_dir):
    os.mkdir(test_stage3_dir)
    
test_stage4_dir = os.path.join(test_dir,'stage4')
if not os.path.exists(test_stage4_dir):
    os.mkdir(test_stage4_dir)


# In[ ]:


mstr_map_df=pd.read_csv(LBL_MAP_FILE)

data_set_lst=[tmp_wrd.replace('', '') for tmp_wrd in os.listdir(DATA_DIR)]
print("*> There are "+str(len(data_set_lst))+" images in \""+DATA_DIR+"\" directory.")

mstr_map_df=pd.read_csv(LBL_MAP_FILE)

lbl_map_df=mstr_map_df[mstr_map_df['image'].isin(data_set_lst)].reset_index(drop=True)

print("*> Label extracted for all the "+str(len(lbl_map_df))+" images from master map DataFrame.")

lbl_map_df=lbl_map_df.set_index(lbl_map_df.columns[0])


# In[ ]:


file_names = [str(idx) for idx,row in lbl_map_df.sample(SAMPL).iterrows()]
for item in file_names:
    sorc = os.path.join(DATA_DIR, item)
    dest = os.path.join(SMP_DATA_DIR, item)
    shutil.copyfile(sorc, dest)


mstr_map_df=pd.read_csv(LBL_MAP_FILE)

data_set_lst=[tmp_wrd.replace('', '') for tmp_wrd in os.listdir(SMP_DATA_DIR)]
print("*> There are "+str(len(data_set_lst))+" images in \""+SMP_DATA_DIR+"\" directory.")

mstr_map_df=pd.read_csv(LBL_MAP_FILE)

lbl_map_df=mstr_map_df[mstr_map_df['image'].isin(data_set_lst)].reset_index(drop=True)

print("*> Label extracted for all the "+str(len(lbl_map_df))+" images from master map DataFrame.")

lbl_map_df=lbl_map_df.set_index(lbl_map_df.columns[0])


# In[ ]:


img_idx = lbl_map_df.index

train_idx, valid_idx = train_test_split(img_idx, test_size = 0.20, random_state = 10, stratify = lbl_map_df['level'])
trn_lbl_map_df = lbl_map_df[lbl_map_df.index.isin(train_idx.tolist())]
vld_lbl_map_df = lbl_map_df[lbl_map_df.index.isin(valid_idx.tolist())]


# In[ ]:


#TRAINING SETUP

#STAGE0
file_names = [str(idx) for idx,row in trn_lbl_map_df[trn_lbl_map_df.level == 0].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(train_stage0_dir, item)
    shutil.copyfile(sorc, dest)
    
#STAGE1
file_names = [str(idx) for idx,row in trn_lbl_map_df[trn_lbl_map_df.level == 1].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(train_stage1_dir, item)
    shutil.copyfile(sorc, dest)
    
#STAGE2
file_names = [str(idx) for idx,row in trn_lbl_map_df[trn_lbl_map_df.level == 2].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(train_stage2_dir, item)
    shutil.copyfile(sorc, dest)
    
#STAGE3
file_names = [str(idx) for idx,row in trn_lbl_map_df[trn_lbl_map_df.level == 3].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(train_stage3_dir, item)
    shutil.copyfile(sorc, dest)

#STAGE4
file_names = [str(idx) for idx,row in trn_lbl_map_df[trn_lbl_map_df.level == 4].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(train_stage4_dir, item)
    shutil.copyfile(sorc, dest)


# In[ ]:


#VALIDATION SETUP

#STAGE0
file_names = [str(idx) for idx,row in vld_lbl_map_df[vld_lbl_map_df.level == 0].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(val_stage0_dir, item)
    shutil.copyfile(sorc, dest)
    
#STAGE1
file_names = [str(idx) for idx,row in vld_lbl_map_df[vld_lbl_map_df.level == 1].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(val_stage1_dir, item)
    shutil.copyfile(sorc, dest)
    
#STAGE2
file_names = [str(idx) for idx,row in vld_lbl_map_df[vld_lbl_map_df.level == 2].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(val_stage2_dir, item)
    shutil.copyfile(sorc, dest)
    
#STAGE3
file_names = [str(idx) for idx,row in vld_lbl_map_df[vld_lbl_map_df.level == 3].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(val_stage3_dir, item)
    shutil.copyfile(sorc, dest)
    
#STAGE4
file_names = [str(idx) for idx,row in vld_lbl_map_df[vld_lbl_map_df.level == 4].iterrows()]
for item in file_names:
    sorc = os.path.join(SMP_DATA_DIR, item)
    dest = os.path.join(val_stage4_dir, item)
    shutil.copyfile(sorc, dest)


# In[ ]:


#CHECKING TRANSFER

print('total training stage0 images:', len(os.listdir(train_stage0_dir)))
print('total validation stage0 images:', len(os.listdir(val_stage0_dir)))


print('total training stage1 images:', len(os.listdir(train_stage1_dir)))
print('total validation stage1 images:', len(os.listdir(val_stage1_dir)))

print('total training stage2 images:', len(os.listdir(train_stage2_dir)))
print('total validation stage2 images:', len(os.listdir(val_stage2_dir)))

print('total training stage3 images:', len(os.listdir(train_stage3_dir)))
print('total validation stage3 images:', len(os.listdir(val_stage3_dir)))

print('total training stage4 images:', len(os.listdir(train_stage4_dir)))
print('total validation stage4 images:', len(os.listdir(val_stage4_dir)))


# In[ ]:



steps_per_epoch = sample_train/batch_size
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(width, height, 3))
conv_base.summary()

print('Conv_Base Summary');


# In[ ]:


batch_size = 32

# this is the augmentation configuration we will use for training
datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = datagen.flow_from_directory(
        train_dir,  
        target_size=(250, 250), 
        batch_size=batch_size,
        class_mode='categorical') 

validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(250, 250),
        batch_size=batch_size,
        class_mode='categorical')


# In[ ]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def AHE(imag):
    img_ahe = exposure.equalize_adapthist(imag, clip_limit=0.30)
    img_grayscale = rgb2gray(img_ahe)
    return img_grayscale

datagen =  ImageDataGenerator(
            zca_whitening=True,
            zoom_range=0.3,
            fill_mode='nearest',
            vertical_flip=True,
            preprocessing_function=AHE)



def get_me_features(direct, samp_cnt):
    features = np.zeros(shape=(samp_cnt, 6, 6, 512))
    labels = np.zeros(shape=(samp_cnt,5)) 
    
    #Because MaxPooling layer in out base model is of shape 6, 6, 512
    
    dat_generator = datagen.flow_from_directory(
                                            direct,
                                            target_size=(width,height),
                                            batch_size = batch_size,
                                            class_mode='categorical')

    idx = 0
    for inputs_batch, labels_batch in dat_generator:
        features_batch = conv_base.predict(inputs_batch)
        #print("+++++++++++++++++++++++++++++++++++++")
        #print(str(features_batch.shape)+"=="+str(features.shape))
        #print(str(labels_batch.shape)+"=="+str(labels.shape))
        #print("+++++++++++++++++++++++++++++++++++++")
        features[idx * batch_size : (idx + 1) * batch_size] = features_batch
        labels[idx * batch_size : (idx + 1) * batch_size] = labels_batch
        idx += 1
        if idx * batch_size >= samp_cnt:
            break
    return features, labels


# In[ ]:


print("\n*> train_features/labels\n")
train_features, train_labels = get_me_features(train_dir, 481)  


# In[ ]:


print("\n*> validation_features/labels\n")
validation_features, validation_labels = get_me_features(validation_dir, 120)


# In[ ]:


print(train_features.shape)


# In[ ]:


epochs = 50

model = models.Sequential()
model.add(layers.Flatten(input_shape=(6,6,512)))
model.add(layers.Dense(256, activation='relu', input_dim=(6*6*512)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='sigmoid'))
model.summary()

# Compile model
model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['acc'])
              
# Train model
#from sklearn.utils import class_weight
#weights = class_weight.compute_sample_weight('balanced', train_labels)


history = model.fit(train_features, train_labels,
                    epochs=epochs,
                    batch_size=batch_size, 
                    validation_data=(validation_features, validation_labels))


# In[ ]:


import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
y_pred1 = model.predict(validation_features)


# In[ ]:





# In[ ]:




