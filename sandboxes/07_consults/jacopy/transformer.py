# %% 
import torch
import torch.nn as nn
from transformers import AutoTokenizer

#class Consultformer(nn.Module):
#    def __init__(self, batch_size : int, num_class : int, tokenizer_path : str, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#
#        self.tokenizer     = AutoTokenizer.from_pretrained(tokenizer_path)
#        self.cls_token     = nn.Parameter(torch.randn((batch_size,1,768)),requires_grad=True)
#
#        self.mlp = []
#        val = 768
#        while val > 3:
#            prev = val
#            succ = val//2
#            self.mlp.append(nn.Linear(prev,succ,bias=False))
#            self.mlp.append(nn.BatchNorm1d(succ))
#            self.mlp.append(nn.ReLU(inplace=True))
#            self.mlp.append(nn.Dropout(0.1))
#            val = succ
#
#        self.mlp.append(nn.Linear(succ,num_class,bias=False))
#
#        self.mlp = nn.Sequential(*self.mlp)
#        
#
#    def forward(self, token_ids : torch.Tensor):
#        
#        with torch.no_grad():
#            token_vector = self.embeddings(token_ids)
#            token_vector = torch.concat([self.cls_token,token_vector],dim=1)
#            self.clf_token = self.encoder(token_vector)[:,0,:] 
#        
#        return self.mlp(self.clf_token)

# %% 
from h_transformer_id import HAttention1D
attn = HAttention1D(dim=768)

# %%
attn_vect = attn(torch.randn((1,5000,768)))
attn_vect.shape
# %%
