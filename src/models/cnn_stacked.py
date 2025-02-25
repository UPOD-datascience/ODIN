import torch
import torch.nn as nn
import torch.nn.functional as F

class VStackedDCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_embedding_dim, maxlen, num_layers, num_classes, dilations, dropout=0.0, pre_trained_embeddings=None, freeze_embeddings=True):
        super(VStackedDCNNClassifier, self).__init__()
        # Trainable embedding layer
        self.trainable_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Pre-trained embedding layer
        if pre_trained_embeddings is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(pre_trained_embeddings, freeze=freeze_embeddings, padding_idx=0)
            self.embedding_output_dim = embedding_dim + pre_embedding_dim
        else:
            self.pretrained_embedding = None
            self.embedding_output_dim = embedding_dim

        self.convs = nn.ModuleList([
            nn.Conv1d(self.embedding_output_dim, num_layers, kernel_size=5, dilation=dilation)
            for dilation in dilations
        ])
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_layers * len(dilations), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        trainable_emb = self.trainable_embedding(x)
        if self.pretrained_embedding is not None:
            pretrained_emb = self.pretrained_embedding(x)
            x = torch.cat([trainable_emb, pretrained_emb], dim=-1)
        else:
            x = trainable_emb

        x = x.permute(0, 2, 1)
        convs_out = [F.relu(conv(x)) for conv in self.convs]
        pooled_out = [self.global_max_pool(conv).squeeze(-1) for conv in convs_out]
        x = torch.cat(pooled_out, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x