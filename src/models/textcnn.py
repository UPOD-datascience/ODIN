'''
TextCNN model from Kim et al.: https://aclanthology.org/D14-1181/


'''

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:14:26 2019

@author: HSU, CHIH-CHAO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Text CNN model
class textCNN(nn.Module):
    
    def __init__(self, vocab_built, dim_channel, kernel_wins, dropout_rate, num_class):
        super(textCNN, self).__init__()
        #load pretrained embedding in embedding layer.
        emb_dim = vocab_built.vectors.size()[1]
        self.embed = nn.Embedding(len(vocab_built), emb_dim)
        self.embed.weight.data.copy_(vocab_built.vectors)
    
        #Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        #Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        #FC layer
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
        
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)
        
        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit
    
class textCNN2(nn.Module):
    def __init__(
        self, vocab_size,embedding_dim, kernel_sizes, num_filters, num_classes, d_prob, mode):
        
        super(textCNN2, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.load_embeddings()
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                             out_channels=num_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = self.embedding(x.T).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        return torch.sigmoid(x).squeeze()

    def load_embeddings(self):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(vocab.vectors)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError('Unexpected value of mode. Please choose from static, nonstatic, rand.')
        