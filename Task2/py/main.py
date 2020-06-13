#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import random
import pandas as pd


# In[4]:


import sys   


# In[10]:


def load_data():    
    df = pd.read_csv('train.csv')
    user_items = {}
    itemId_max=[]
    for i,row in df.iterrows():
        user = int(row[0])
        user_items[user] = [int(x) for x in row[1].split()]
        itemId_max.append(max(user_items[user]))
    num_users = max(user_items.keys())+1
    num_items = max(itemId_max)+1
    return num_users, num_items, user_items


# In[11]:


# load train.csv
user_size, item_size,user_items = load_data()


# In[12]:


model = torch.load('bpr.pth')


# In[13]:


w = list(model.parameters())
user = w[0].detach().cpu().numpy()
item = w[1].detach().cpu().numpy()
interaction = np.dot(user,item.T)


# In[14]:


predict = pd.DataFrame(columns=['UserId','ItemId'])
for uid, items in enumerate(interaction):
    for i in user_items[uid]:
        items[i] = -99
    topk = np.argsort(-items)[:50]
    predict.loc[uid,'UserId'] = uid
    predict.loc[uid,'ItemId'] = ' '.join([str(x) for x in topk])


# In[ ]:


output = sys.argv[1]


# In[6]:


predict.to_csv( output ,index = 0)

