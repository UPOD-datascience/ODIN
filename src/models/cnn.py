import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_embedding_dim, maxlen, num_layers, num_classes, kernel_size=5, dropout=0.0, pre_trained_embeddings=None, freeze_embeddings=True):
        super(CNNClassifier, self).__init__()
        # Trainable embedding layer
        self.trainable_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Pre-trained embedding layer
        if pre_trained_embeddings is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(pre_trained_embeddings, freeze=freeze_embeddings, padding_idx=0)
            self.embedding_output_dim = embedding_dim + pre_embedding_dim
        else:
            self.pretrained_embedding = None
            self.embedding_output_dim = embedding_dim

        self.conv = nn.Conv1d(self.embedding_output_dim, num_layers, kernel_size=kernel_size)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_layers, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        trainable_emb = self.trainable_embedding(x)
        if self.pretrained_embedding is not None:
            pretrained_emb = self.pretrained_embedding(x)
            x = torch.cat([trainable_emb, pretrained_emb], dim=-1)
        else:
            x = trainable_emb

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = self.global_max_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class CNNClassifierWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_embedding_dim, maxlen, num_layers, num_classes, kernel_size=5, dropout=0.0, pre_trained_embeddings=None, freeze_embeddings=True):
        super(CNNClassifier, self).__init__()
        # Trainable embedding layer
        self.trainable_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Pre-trained embedding layer
        if pre_trained_embeddings is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(pre_trained_embeddings, freeze=freeze_embeddings, padding_idx=0)
            self.embedding_output_dim = embedding_dim + pre_embedding_dim
        else:
            self.pretrained_embedding = None
            self.embedding_output_dim = embedding_dim

        self.conv = nn.Conv1d(self.embedding_output_dim, num_layers, kernel_size=kernel_size)

        # Attention mechanism
        self.attention_fc = nn.Linear(num_layers * 2, 1)  # Calculate attention weights
        self.attention_softmax = nn.Softmax(dim=1)  # Normalize attention weights

        self.fc = nn.Linear(num_layers, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        trainable_emb = self.trainable_embedding(x)
        if self.pretrained_embedding is not None:
            pretrained_emb = self.pretrained_embedding(x)
            x = torch.cat([trainable_emb, pretrained_emb], dim=-1)
        else:
            x = trainable_emb

        # Attention mechanism
        attention_scores = self.attention_fc(x)  # (batch_size, seq_len, 1)
        attention_weights = self.attention_softmax(attention_scores.squeeze(-1))  # (batch_size, seq_len)
        context_vector = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # Weighted sum (batch_size, num_layers * 2)

        # Dropout and fully connected layer
        x = self.dropout(context_vector)
        x = self.fc(x)
        return x
    