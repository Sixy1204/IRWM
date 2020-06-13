#!/usr/bin/env python
# coding: utf-8

# In[52]:


from torch.utils.data import Dataset
import random
import numpy as np

# In[53]:


class Data(Dataset):
    def __init__(self, user_size, item_size, user_items):
        self.user_size = user_size
        self.item_size = item_size
        self.user_items = user_items
        self.test_pos = self.test_pos()
        self.train_list, self.train_pair = self.train()
        
    def get_neg(self):  
        self.input_data = []
        for u, i in self.train_pair:
            j = np.random.randint(self.item_size)
        
            while j in self.train_list[u]:
                j = np.random.randint(self.item_size)
            self.input_data.append([u, i, j]) 

    def __getitem__(self, index):
        
        u, i, j = self.input_data[index]
        return u, i, j
    
    def __len__(self): 
        return len(self.train_pair)
    
    def test_pos(self):
        user_test = {}
        for u,i_list in self.user_items.items():
            user_test[u] = random.sample(self.user_items[u],1)[0]
        return user_test

    def train(self):   
        train_lst = {}
        pair = []
        for uid, items in self.user_items.items():
            train_i = []
            for i in items:
                train_i.append(i)
                pair.append((uid,i))
            train_lst[uid] = train_i
        return train_lst,pair

    

