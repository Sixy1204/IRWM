#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn


# In[4]:


class BPR(nn.Module):
    def __init__(self, user_size, item_size, embedding_size, batch_size, device):
        super(BPR, self).__init__()
        self.user_matrix = nn.Embedding(user_size, embedding_size)
        self.item_matrix = nn.Embedding(item_size, embedding_size)
        
        nn.init.normal_(self.user_matrix.weight, std=0.01)
        nn.init.normal_(self.item_matrix.weight, std=0.01)
        
        self.batch = batch_size
        self.device = device

    def forward(self, u, i, j):

        ui = torch.mul(self.user_matrix(u), self.item_matrix(i)).sum(dim=1)
        uj = torch.mul(self.user_matrix(u), self.item_matrix(j)).sum(dim=1)
        
        loss = -torch.log(torch.sigmoid(ui - uj)).sum()
        
        return loss

