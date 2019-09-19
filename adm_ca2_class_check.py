#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pandas as pd
import seaborn as seab

#Project Configuration
PROJ_WD='C:/Users/darsh/Documents/SEM2/ADM/ADM/Project'
DATA_DIR=PROJ_WD+'/full_train'
LBL_MAP_FILE=PROJ_WD+'/trainLabels.csv'
DATA_DIR_LD=PROJ_WD+'/train_cls'


# In[10]:


data_set_lst=[tmp_wrd.replace('', '') for tmp_wrd in os.listdir(DATA_DIR)]
print("*> There are "+str(len(data_set_lst))+" images in \""+DATA_DIR+"\" directory.")


# In[12]:


mstr_map_df=pd.read_csv(LBL_MAP_FILE)

mstr_map_df['image'] = mstr_map_df['image'].astype(str)

lbl_map_df=mstr_map_df[mstr_map_df['image'].isin(data_set_lst)].reset_index(drop=True)

print("*> Label extracted for all the "+str(len(lbl_map_df))+" images from master map DataFrame.")


# In[13]:


# This was to make sure that we don't have any pationt's id just with one side eye image (left or right)

pat_count = []
for idx,lvl in lbl_map_df.iterrows():
    if lvl[0] in mstr_map_df['image'].tolist():
        pat_count.append(lvl[0].split('_')[0])

pat_count=pd.Series(pat_count)
pat_freq=pat_count.value_counts()
pat_freq[pat_freq != 2]


# In[14]:


patients_to_drop=lbl_map_df

patients_to_drop['image']=patients_to_drop['image'].str.replace('_left.jpeg','')
patients_to_drop['image']=patients_to_drop['image'].str.replace('_right.jpeg','')

patients_to_drop=patients_to_drop.groupby('image').sum()

patients_to_drop_lst=[]
for idx, row in patients_to_drop.iterrows():
    if int(row[0])%2 != 0:
        patients_to_drop_lst.append(str(idx)+"_right.jpeg")
        patients_to_drop_lst.append(str(idx)+"_left.jpeg")
    
print("*> From master data frame there are "+str(mstr_map_df[mstr_map_df['image'].isin(patients_to_drop_lst)].count()[0])+" images not having both eye stage same.")
mstr_map_df=mstr_map_df[~mstr_map_df['image'].isin(patients_to_drop_lst)]

print("*> Data Frame after cleaning: "+str(mstr_map_df.shape))


# In[17]:


df_gaussian_sampling = lbl_map_df.sample(n=5000)


# In[18]:


# 0=Normal
# 1=Mild DR
# 2=Moderate DR
# 3=Severe DR
# 4=Proliferative DR

seab.countplot(x="level", data=df_gaussian_sampling)

