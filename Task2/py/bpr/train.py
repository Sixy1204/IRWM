#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler


# In[3]:


import numpy as np
import random
import pandas as pd



# In[5]:


from Dataset import Data
from bprModel import BPR


# In[6]:


def load_data():    
    df = pd.read_csv('./bpr/train.csv')
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


def train(loader, model, optimizer, epochs, batch_size, device):
    trainLoss = []
    valLoss = []
    for epoch in range(epochs+1):
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
                    
        avg_train_batch_loss = torch.mean(torch.FloatTensor(train_loss))/batch_size
        avg_val_batch_loss = torch.mean(torch.FloatTensor(val_loss))/batch_size
        
        trainLoss.append(avg_train_batch_loss)
        valLoss.append(avg_val_batch_loss)
        
        print(f"Epoch : {epoch} | Avg. train batch loss = {avg_train_batch_loss:.4f} | Avg. val batch loss = {avg_val_batch_loss:.4f}\n")
    
    #return trainLoss, valLoss


# In[8]:


user_size, item_size,user_items = load_data()


# In[9]:


batch_size = 3000
epochs = 70
embedding_size = 128


# In[10]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# In[11]:


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


# In[12]:


model = BPR(user_size, item_size, embedding_size, batch_size, device)


# In[13]:


optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


# In[14]:


train(data_loaders, model, optimizer, epochs, batch_size, device)


# In[16]:


torch.save(model, './bpr/bpr_model.pth')

