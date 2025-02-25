import torch
import torch.nn as nn
import torch.nn.functional as F


'''
# Example pre-trained embeddings
vocab_size = 50000
embedding_dim = 128
pre_embedding_dim = 300
pretrained_embeddings = torch.rand(vocab_size, pre_embedding_dim)

# RCNNClassifier
rcnn_model = RCNNClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    pre_embedding_dim=pre_embedding_dim,
    maxlen=256,
    num_layers=128,
    num_classes=5,
    dropout=0.5,
    dilation=4,
    pre_trained_embeddings=pretrained_embeddings,
    freeze_embeddings=True
)

# RCNNVStackedClassifier
rcnn_vstacked_model = RCNNVStackedClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    pre_embedding_dim=pre_embedding_dim,
    maxlen=256,
    num_layers=128,
    num_classes=5,
    dropout=0.5,
    dilation=4,
    pre_trained_embeddings=pretrained_embeddings,
    freeze_embeddings=True
)

# Example input
input_data = torch.randint(0, vocab_size, (16, 256))
output = rcnn_model(input_data)
print(output.shape)  # (16, 5)

output = rcnn_vstacked_model(input_data)
print(output.shape)  # (16, 5)

'''

class RCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_embedding_dim, maxlen, num_layers, num_classes, dropout=0.0, dilation=4, pre_trained_embeddings=None, freeze_embeddings=True):
        super(RCNNClassifier, self).__init__()
        # Trainable embedding layer
        self.trainable_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Pre-trained embedding layer
        if pre_trained_embeddings is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(pre_trained_embeddings, freeze=freeze_embeddings, padding_idx=0)
            self.embedding_output_dim = embedding_dim + pre_embedding_dim
        else:
            self.pretrained_embedding = None
            self.embedding_output_dim = embedding_dim

        # Layers
        self.bigru = nn.GRU(self.embedding_output_dim, num_layers, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(num_layers * 2, num_layers, kernel_size=5, dilation=dilation)
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

        x, _ = self.bigru(x)
        x = x.permute(0, 2, 1)  # Prepare for Conv1D
        x = F.relu(self.conv(x))
        x = self.global_max_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class RCNNVStackedClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_embedding_dim, maxlen, num_layers, num_classes, dropout=0.0, dilation=4, pre_trained_embeddings=None, freeze_embeddings=True):
        super(RCNNVStackedClassifier, self).__init__()
        # Trainable embedding layer
        self.trainable_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Pre-trained embedding layer
        if pre_trained_embeddings is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(pre_trained_embeddings, freeze=freeze_embeddings, padding_idx=0)
            self.embedding_output_dim = embedding_dim + pre_embedding_dim
        else:
            self.pretrained_embedding = None
            self.embedding_output_dim = embedding_dim

        # Layers
        self.bigru = nn.GRU(self.embedding_output_dim, num_layers, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(self.embedding_output_dim, num_layers, kernel_size=5, dilation=dilation)
        self.global_max_pool_bigru = nn.AdaptiveMaxPool1d(1)
        self.global_max_pool_conv = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_layers * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        trainable_emb = self.trainable_embedding(x)
        if self.pretrained_embedding is not None:
            pretrained_emb = self.pretrained_embedding(x)
            x = torch.cat([trainable_emb, pretrained_emb], dim=-1)
        else:
            x = trainable_emb

        # BiGRU
        bigru_out, _ = self.bigru(x)
        bigru_out = bigru_out.permute(0, 2, 1)
        bigru_pooled = self.global_max_pool_bigru(bigru_out).squeeze(-1)

        # Conv1D
        conv_out = x.permute(0, 2, 1)  # Prepare for Conv1D
        conv_out = F.relu(self.conv(conv_out))
        conv_pooled = self.global_max_pool_conv(conv_out).squeeze(-1)

        # Concatenate and pass through the classifier
        concat_out = torch.cat([bigru_pooled, conv_pooled], dim=-1)
        x = self.dropout(concat_out)
        x = self.fc(x)
        return x
