from typing import List
import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from functions.h_transformer_1d import HAttention1D, FeedForward, RotaryEmbedding

from jacopy.resnet1d_v2 import ResNet1D


class ResConsultNet(nn.Module):
    def __init__(self, 
                 tokenizer_vocab_path  : str       = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/vocab.json',
                 tokenizer_merges_path : str       = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/merges.txt',
                 vocab_size            : int       = 5000,
                 padding_size          : int       = 5000,
                 embedding_dim         : int       = 512,
                 resnet_layers         : List[int] = [2,2,2,2,2],
                 base_filters          : int       = 32,
                 pool_size             : int       = 2,
                 num_class             : int       = 2,
                 device                : str       = 'cuda',
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
        self.device       = device

        t = ByteLevelBPETokenizer(
            vocab =tokenizer_vocab_path ,
            merges=tokenizer_merges_path,
            add_prefix_space = True,
            )
        
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=t._tokenizer)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.embeddings = nn.Embedding(num_embeddings = vocab_size +1,
                                       embedding_dim  = embedding_dim)
        
        self.resnet     = ResNet1D(layers       = resnet_layers,
                                   base_filters = base_filters,
                                   pool_size    = pool_size,
                                   num_class    = num_class)
        

    def forward(self, text : str):
        
        with torch.no_grad():
            
            encoded_text = self.tokenizer(
                text,
                return_attention_mask=False,
                return_length=False,
                return_token_type_ids=False,
                padding='max_length',
                max_length = self.padding_size,
                return_tensors='pt',
                truncation=True
            )['input_ids']
        # print(f'Tokenizing completed... {encoded_text.shape}')

        embedded_text = self.embeddings(encoded_text.to(self.device))
        
        # print(f'Embedding completed... {embedded_text.shape}')
        
        out = self.resnet(embedded_text)

        # print(f'ResNet output: {out.shape}')

        return out
    


class ConsultFormer(nn.Module):
    """
    David's Consult Transformer
    
    Parameters:
    - N (number of iterations of the self attention algorithm), Default: 5
    - heads (number of feature maps), Default: 8
    - device (type of device: CPU or GPU), Default: cuda
    - ff_mult (multiplical factor of the next layer in respect of the current layer), Default: 4
    - padding_size (max length of a sentence), Default: 5000
    - embedding_dim (length of each embedding vector), Default: 512
    - num_embeddings (length of the embeddings dictionary), Default: 5001
    - tokenizer_vocab_path
    - tokenizer_merges_path
    """

    def __init__(
            self, 
            N                       : int = 5,
            heads                   : int = 8,
            device                  : str = 'cuda',
            ff_mult                 : int = 4,
            padding_size            : int = 5000,
            embedding_dim           : int = 512,
            num_embeddings          : int = 5001,
            tokenizer_vocab_path    : str = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/vocab.json', 
            tokenizer_merges_path   : str = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/merges.txt', 
            *args,
            **kwargs
        ):
        
        super().__init__(*args, **kwargs)
        
        assert embedding_dim % heads == 0 , "The embedding size is not divisible by the number of heads"
        
        self.device = device
        
        self.padding_size = padding_size
        
        # Inizializzazione del tokenizer che effettua una
        # tokenizzazione basata su Byte Pair Encoding (BPE),
        # algoritmo di tokenizzazione che unisce coppie di byte o 
        # caratteri più frequenti in sequenze. In questo modo è 
        # in grado di creare nuovi tokens basati sulla frequenza delle
        # coppie di caratteri nei dati di addestramento. Questo approccio
        # aiuta a gestire parole rare o sconosciute creando token basati
        # su parti comuni di parole
        t = ByteLevelBPETokenizer(
            vocab = tokenizer_vocab_path,
            merges = tokenizer_merges_path,
            add_prefix_space = True,
        )
        
        # Creazione della classe di Hugging Face che fornisce una versione 
        # veloce del tokenizer, capace di applicare la tokenizzazione in 
        # modo molto più rapido rispetto alle versioni standard
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object = t._tokenizer)
        
        # Viene aggiunto il token di padding al dizionario dei token speciali
        # del tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Creazione del layer di embedding
        self.embedder = nn.Embedding(
            device = device, 
            embedding_dim = embedding_dim,
            num_embeddings = num_embeddings
        )
        
        # Definizione del positional embedding
        self.pos_emb = RotaryEmbedding(dim = self.dim_head)
        #self.pos_enc = nn.Parameter(data=torch.randn([1, embedding_dim]))
        
        # Creazione dell'encoder in cui si avrà una sequenza 
        # di encoder layer (attention + feed forward) ripetuta per N volte
        self.encoder = nn.Sequential()
        
        for _ in range(N):
            self.encoder.append(
        
            )
            
            self.encoder.append(
                FeedForward(embedding_dim, mult = ff_mult)
            )
    
    def forward(self, text):
        
        with torch.no_grad():
            
            txt2ids = self.tokenizer(
                text,
                return_attention_mask=False,
                return_length=False,
                return_token_type_ids=False,
                padding='max_length',
                max_length = self.padding_size,
                return_tensors='pt',
                truncation=True
                )['input_ids']
        
        embeddings = self.embedder(txt2ids)
        
        
        
        
        
        
        pass


class AddNorm_HAttention(nn.Module):
    
    def __init__(self, 
                embedding_dim,
                dim_head, 
                heads, 
                block_size,
                pos_emb):
        
        super().__init__()
        
        self.attention  = HAttention1D(
            dim         = embedding_dim, 
            dim_head    = dim_head, 
            heads       = heads, 
            block_size  = block_size,
            pos_emb     = pos_emb 
        )
        
        self.lnorm = nn.LayerNorm()





if __name__ == '__main__':

    net = ResConsultNet(
        tokenizer_vocab_path ='sandboxes/07_consults/pretrained/DutchEHRTokenizer/vocab.json',
        tokenizer_merges_path='sandboxes/07_consults/pretrained/DutchEHRTokenizer/merges.txt',
                        
                        )

    print(net(['hoi ik ben jacopo ik ben 28 jaar oud en ik kom uit italie',
               'pizza is het beste in de wereld']))