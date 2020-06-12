#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler


# In[4]:


import numpy as np
import os
import random
import pandas as pd


# In[5]:


from Dataset import Data
from bprModel import BPR


# In[6]:


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


# In[7]:


user_size, item_size,user_items = load_data()


# In[52]:


batch_size = 3000
epochs = 70
embedding_size = 128


# In[9]:


dataset = Data(user_size, item_size, user_items)


# In[10]:


print('The number of train pairs is %d'%len(dataset.train_pair))


# In[11]:


loader = DataLoader(dataset, batch_size, shuffle=True)


# In[25]:


dataset = Data(user_size, item_size, user_items)
validation_split = 0.1
shuffle_dataset = True

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(233)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
data_loaders = {"train": train_loader, "val": validation_loader}
data_lengths = {"train": len(train_indices), "val": len(val_indices)}


# In[40]:


def train(loader, model, optimizer, epochs, batch_size, device):

    #total_loss = 0.0
    #batch_count = 0

    for epoch in range(epochs):
        train_loss = []
        val_loss = []
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  
            else:
                model.train(False)  
            
            model.to(device)
            data_loaders[phase].dataset.get_neg()
            for batch, (batch_u, batch_i, batch_j) in enumerate(data_loaders[phase]):
                
                batch_u = batch_u.to(device)
                batch_i = batch_i.to(device)
                batch_j = batch_j.to(device)
            
                loss = model(batch_u, batch_i, batch_j)
                
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.data)
                else:
                    val_loss.append(loss.data)
                
                #batch_count += 1
                #total_loss += loss.data

                #avg_loss = total_loss / batch_count
            
            
        print(f"Training Epoch : {epoch} | Train Loss = {np.mean(train_loss)/batch_size:.4f} | Val Loss = {np.mean(val_loss)/batch_size:.4f}\n")
                       


# In[13]:


def _eval(model, test_pos, test_sample):
    
    model.eval()
    result = model.predict(test_sample)
    num_users = result.shape[0]

    hit = 0
    ndcg = 0

    for i in range(num_users):
        
        retrieve_items = list(result[i])
        label = test_pos[i]

        if label in retrieve_items:
            hit += 1
            ndcg += (1 / math.log(retrieve_items.index(label)+2,2))

    return (hit / num_users), (ndcg / num_users)


# In[18]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# In[19]:


model = BPR(user_size, item_size, embedding_size, batch_size, device)


# In[20]:


optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


# In[ ]:


train(loader, model, optimizer, epochs, batch_size, device)


# In[42]:


w = list(model.parameters())


# In[43]:


user = w[0].detach().numpy()


# In[44]:


item =  w[1].detach().numpy()


# In[45]:


interaction = np.dot(user,item.T)


# In[46]:


predict = pd.DataFrame(columns=['UserId','ItemId'])


# In[47]:


for uid, items in enumerate(interaction):
    for i in user_items[uid]:
        items[i] = -99
    topk = np.argsort(-items)[:50]
    predict.loc[uid,'UserId'] = uid
    predict.loc[uid,'ItemId'] = ' '.join([str(x) for x in topk])


# In[48]:


predict.iloc[0]


# In[49]:


predict.shape


# In[50]:


predict.to_csv('submit_612.csv',index = 0)

