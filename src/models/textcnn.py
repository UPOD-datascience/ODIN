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


class TextCNNClassifierWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_embedding_dim, maxlen, num_layers, num_classes, filters, dropout=0.0, pre_trained_embeddings=None, freeze_embeddings=True):
        super(TextCNNClassifierWithAttention, self).__init__()
        # Trainable embedding layer
        self.trainable_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Pre-trained embedding layer
        if pre_trained_embeddings is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(pre_trained_embeddings, freeze=freeze_embeddings, padding_idx=0)
            self.embedding_output_dim = embedding_dim + pre_embedding_dim
        else:
            self.pretrained_embedding = None
            self.embedding_output_dim = embedding_dim

        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_layers, (filter_size, self.embedding_output_dim))
            for filter_size in filters
        ])
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))  # Explicit pooling layer

        # Attention mechanism
        self.attention_fc = nn.Linear(num_layers * len(filters), num_layers * len(filters))
        self.attention_softmax = nn.Softmax(dim=1)

        # Fully connected layer
        self.fc = nn.Linear(num_layers * len(filters), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        trainable_emb = self.trainable_embedding(x)
        if self.pretrained_embedding is not None:
            pretrained_emb = self.pretrained_embedding(x)
            x = torch.cat([trainable_emb, pretrained_emb], dim=-1)
        else:
            x = trainable_emb

        x = x.unsqueeze(1)  # Add channel dimension for Conv2D
        convs_out = [F.relu(conv(x)) for conv in self.convs]
        pooled_out = [self.global_max_pool(conv).squeeze(-1).squeeze(-1) for conv in convs_out]

        # Concatenate all pooled outputs
        concat_out = torch.cat(pooled_out, dim=1)

        # Apply attention mechanism
        attention_weights = self.attention_fc(concat_out)
        attention_weights = self.attention_softmax(attention_weights)
        attended_out = concat_out * attention_weights

        # Pass through dropout and fully connected layer
        x = self.dropout(attended_out)
        x = self.fc(x)
        return x

