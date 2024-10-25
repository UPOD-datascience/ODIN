'''
This file needs to train a custom model defined in jacopy/model.py, notebook starting to be tedious
'''

from jacopy.data         import EHR
from jacopy.resnet1d_v2  import ResNet1D
from jacopy.runner       import train,validate,test
from torch.utils.data    import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
def main():
    
    writer = SummaryWriter(log_dir=f'./sandboxes/07_consults/experiments/{dt.now().strftime("%Y%m%d%H%M%S")}')

    root = 'C:\\Users\\jvitale\\text_dataset'

    dataset_train = EHR(root,'train')
    dataset_test  = EHR(root,'test' )

    '''--- Hyperparameters ---'''
    batch_size    = 3
    num_class     = 2
    epochs        = 20
    learning_rate = 3e-4

    train_loader = DataLoader(dataset_train,batch_size=batch_size   ,shuffle=True ,pin_memory=True)
    #valid_loader = DataLoader(dataset_valid,batch_size=batch_size*10,shuffle=False,pin_memory=True)
    test_loader  = DataLoader(dataset_test ,batch_size=batch_size*10,shuffle=False,pin_memory=True)

    model = ResNet1D(layers       = [2,2,2,2,],
                     base_filters = 32,
                     num_class    = num_class,
                     pool_size    = 2)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    print(model)


    for epoch in range(1,epochs+1):
        train(model,train_loader,optimizer,criterion,'cpu',epoch,writer)
        #validate(model,valid_loader,'cpu',epoch,writer)

    test(model,test_loader,'cpu',1,writer)
    




if __name__ == '__main__':
    main()