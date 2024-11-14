'''
This file needs to train a custom model defined in jacopy/model.py, notebook starting to be tedious
'''

from jacopy.data      import EHR
from jacopy.model     import ResConsultNet
from jacopy.runner    import train,validate,test
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import torch
import os,sys

def main():
    
    writer = SummaryWriter(log_dir=f'./sandboxes/07_consults/experiments/{dt.now().strftime("%Y%m%d%H%M%S")}')
    os.makedirs(os.path.join(writer.log_dir,'checkpoint'))
    root = 'C:\\Users\\jvitale\\data\\whole_text_dataset'

    dataset_train = EHR(root,'train')
    dataset_test  = EHR(root,'test' )

    '''--- Hyperparameters ---'''
    hyperparameters = {
        'batch_size'    : 256,
        'num_class'     : 2,
        'epochs'        : 100,
        'learning_rate' : 3e-4
    }

    train_loader = DataLoader(dataset_train,batch_size=hyperparameters['batch_size']   ,shuffle=True ,pin_memory=True)
    #valid_loader = DataLoader(dataset_valid,batch_size=batch_size*10,shuffle=False,pin_memory=True)
    test_loader  = DataLoader(dataset_test ,batch_size=hyperparameters['batch_size']*10,shuffle=False,pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Devo inserire il modello ConsultFormer una volta finito
    model = ResConsultNet(resnet_layers=[2,2,2,2],
                          pool_size=4,
                          device='cpu',
                          base_filters=16)
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

    criterion = nn.CrossEntropyLoss()

    print(model)

    for epoch in range(1, hyperparameters['epochs'] + 1):
        train(model,train_loader,optimizer,criterion,device,epoch,writer)
        test(model,test_loader,device,epoch,writer)

    test(model,test_loader,device,epoch+1,writer)
    




if __name__ == '__main__':
    main()
