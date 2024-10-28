from typing import List
import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from resnet1d_v2 import ResNet1D
import os

class ResConsultNet(nn.Module):
    def __init__(self, 
                 tokenizer_vocab_path  : str       = r'pretrained\DutchEHRTokenizer\vocab.json',
                 tokenizer_merges_path : str       = r'pretrained\DutchEHRTokenizer\merges.txt',
                 vocab_size            : int       = 5000,
                 padding_size          : int       = 5000,
                 embedding_dim         : int       = 512,
                 resnet_layers         : List[int] = [2,2,2,2,2],
                 base_filters          : int       = 32,
                 pool_size             : int       = 2,
                 num_class             : int       = 2,
                 *args, **kwargs):
        
        '''
        This is the ResConsultNet.

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
        self.padding_size = padding_size

        t = ByteLevelBPETokenizer(
            vocab =tokenizer_vocab_path ,
            merges=tokenizer_merges_path,
            add_prefix_space = True,
            )
        
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=t)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.embeddings = nn.Embedding(num_embeddings = vocab_size +1,
                                       embedding_dim  = embedding_dim)
        
        self.resnet     = ResNet1D(layers       = resnet_layers,
                                   base_filters = base_filters,
                                   pool_size    = pool_size,
                                   num_class    = num_class)
        

    def forward(self, text : str):
        
        print(f'Processing input...')
        
        with torch.no_grad():
            
            encoded_text = self.tokenizer(text,
                                          return_attention_mask=False,
                                          return_length=False,
                                          return_token_type_ids=False,
                                          padding='max_length',
                                          max_length = self.padding_size,
                                          return_tensors='pt',
                                          padding_side='right')['input_ids']
            
        print(f'Tokenizing completed... {encoded_text.shape}')

        embedded_text = self.embeddings(encoded_text)
        
        print(f'Embedding completed... {embedded_text.shape}')

        return self.resnet(embedded_text)
    



class ConsultFormer(nn.Module):
    '''
    This is the David Consult Transformer
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)







if __name__ == '__main__':

    net = ResConsultNet(
        tokenizer_vocab_path ='sandboxes/07_consults/pretrained/DutchEHRTokenizer/vocab.json',
        tokenizer_merges_path='sandboxes/07_consults/pretrained/DutchEHRTokenizer/merges.txt',
                        
                        )

    print(net(['hoi ik ben jacopo ik ben 28 jaar oud en ik kom uit italie',
               'pizza is het beste in de wereld']))