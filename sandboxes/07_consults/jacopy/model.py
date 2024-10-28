from typing import List
import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
from resnet1d_v2 import ResNet1D

class ResConsultNet(nn.Module):
    def __init__(self, 
                 tokenizer_vocab_path  : str       = r'pretrained\DutchEHRTokenizer\vocab.json',
                 tokenizer_merges_path : str       = r'pretrained\DutchEHRTokenizer\merges.txt',
                 vocab_size            : int       = 5000,
                 embedding_dim         : int       = 512,
                 resnet_layers         : List[int] = [2,2,2,2,2],
                 base_filters          : int       = 32,
                 pool_size             : int       = 2,
                 *args, **kwargs):
        
        '''
        This is the ResConsultFormer.

        Parameters
        ---
        - tokenizer_vocab_path  : path/to/vocab.json
        - tokenizer_merges_path : path/to/merges.txt
        - vocab_size            : vocabulary size of the pretrained tokenizer
        - embedding_dim         : embeddig dimension
        - resnet_layers         : resnet layer, list of how many resnet basic block the more the deeper
        - base_filters          : starting filters, filter double along network depth
        - pool_size             : Max pooling size, default 2
        '''

        super().__init__(*args, **kwargs)

        self.tokenizer = ByteLevelBPETokenizer(
            vocab =tokenizer_vocab_path ,
            merges=tokenizer_merges_path,
            add_prefix_space = True,
            )
        
        self.embeddings = nn.Embedding(num_embeddings = vocab_size,
                                       embedding_dim  = embedding_dim)
        
        self.resnet     = ResNet1D(layers       = resnet_layers,
                                   base_filters = base_filters,
                                   pool_size    = pool_size)
        

    def forward(self, text : str):
        with torch.no_grad():
            self.tokenizer.encode(text)