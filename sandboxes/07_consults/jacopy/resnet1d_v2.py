
import torch
import torch.nn as nn

  
class Block(nn.Module):
    def __init__(self, out_channels,kernel_size, is_first,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.is_first = is_first

        self.conv = nn.LazyConv1d(out_channels = out_channels,
                                  kernel_size  = kernel_size,
                                  padding      = 'same')
        
        self.batchn = nn.BatchNorm1d(out_channels)
        self.act    = nn.ReLU()
        self.do     = nn.Dropout1d(0.3)

    def forward(self, x:torch.Tensor):
        if self.is_first:
            return self.act(self.batchn(self.conv(x)))
        else:
            res = x
            return self.act(self.batchn(self.conv(x))) + res
    

        
class ResNet1D(nn.Module):
    def __init__(self, 
                 layers       : list[int], 
                 base_filters : int, 
                 num_class    : int,
                 pool_size    : int, 
                 *args, **kwargs) -> None:
        
        super(ResNet1D, self).__init__(*args, **kwargs)
        self.emb = nn.Embedding(52000,768)
        self.pe = nn.Parameter(torch.random(1,768))
        self.backbone = nn.Sequential()

        for l,layer in enumerate(layers):
            for i in range(layer):
                self.backbone.append(Block(out_channels = base_filters*(2**l),
                                           kernel_size  = 3,
                                           is_first     = i==0,
                                           ))
            
            if l == len(layers)-1:
                continue

            self.backbone.append(nn.MaxPool1d(pool_size))

        self.backbone.append(nn.Flatten())

        self.neural_net = nn.Sequential(nn.LazyLinear(num_class*10),
                                        nn.BatchNorm1d(num_class*10),
                                        nn.ReLU(),
                                        nn.Dropout1d(p=0.5),
                                        nn.LazyLinear(num_class),
        )

        
    def forward(self, x):
        emb = self.emb(x)
        emb = emb + self.pe
        out = self.backbone(x)
        print(out.shape)
        out = self.neural_net(out)
        
        return out
    

if __name__ == "__main__":
    net = ResNet1D(layers       = [2,1,2,1,2,1],
                  base_filters  = 64,
                  num_class     = 4,
                  pool_size     = 2)
    

    
    print(net)

    # The input of the net is always is a 3D array: (batch_size, time_steps, seq_len)
            #  - batch_size: Il numero di sequenze di video nel batch (ad esempio, 1 per un singolo video).
            #  - input_size: il numero di features (ad esempio, 20 coordinate: 10 keypoints x 2 coordinate ciascuno).
            #  - time_steps: Il numero di frame per video (ad esempio, 20).


    input = torch.randn([3, 22, 40])
    
    output = net(input)
    print("***************************************************")
    print(f"La dimensione dell'output è: {output.shape}")