from typing import List

import torch.nn as nn

from functions.resnet_1d import ResNet1D


class ResConsultNet(nn.Module):
    def __init__(
            self, 
            vocab_size            : int       = 5000,
            padding_size          : int       = 5000,
            embedding_dim         : int       = 512,
            resnet_layers         : List[int] = [2,2,2,2,2],
            base_filters          : int       = 32,
            pool_size             : int       = 2,
            num_classes           : int       = 2,
            device                : str       = 'cuda',
            *args, 
            **kwargs
        ):
        
        '''
        This is the ResConsultNet.
        
        Parameters
        ---
        - vocab_size            : vocabulary size of the pretrained tokenizer
        - embedding_dim         : embeddig dimension
        - resnet_layers         : resnet layer, list of how many resnet basic block the more the deeper
        - base_filters          : starting filters, filter double along network depth
        - pool_size             : Max pooling size, default 2
        - num_classes           : Number of classes, default 2
        - device                : default 'cuda'
        '''
        
        super().__init__(*args, **kwargs)
        
        self.padding_size = padding_size
        self.device = device
        
        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size + 1,
            embedding_dim  = embedding_dim
        ).to(device)
        
        self.resnet = ResNet1D(
            layers = resnet_layers,
            base_filters = base_filters,
            pool_size = pool_size,
            num_classes = num_classes,
            device = device
        )
    
    def forward(self, x):
        x = x.to(self.device)
        
        embedded_text = self.embeddings(x)
        
        # print(f'Embedding shape: {embedded_text.shape}')
        
        outputs = self.resnet(embedded_text)
        
        # print(f'ResNet output: {outputs.shape}')
        
        return outputs



if __name__ == '__main__':
    
    net = ResConsultNet(
        vocab_size      = 5000,
        padding_size    = 5000,
        embedding_dim   = 512,
        resnet_layers   = [2,2,2,2,2],
        base_filters    = 32,
        pool_size       = 2,
        num_classes     = 2,
        device          = 'cpu'
    )
    
    print(net([
        'hoi ik ben jacopo ik ben 28 jaar oud en ik kom uit italie',
        'pizza is het beste in de wereld'
    ]))