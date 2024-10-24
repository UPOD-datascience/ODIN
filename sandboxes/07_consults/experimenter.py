'''
This file needs to train a custom model defined in jacopy/model.py, notebook starting to be tedious
'''

from jacopy.data   import EHR
from jacopy.model  import Consultformer
from jacopy.runner import train,validate,test
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
from torchsummary import summary
import torch.optim as optim
def main():
    
    writer = SummaryWriter(log_dir=f'./sandboxes/07_consults/experiments/{dt.now().strftime("%Y%m%d%H%M%S")}')

    root = 'C:\\Users\\jvitale\\text_dataset'

    dataset_train = EHR(root,'train')
    dataset_valid = EHR(root,'valid')
    dataset_test  = EHR(root,'test' )

    '''--- Hyperparameters ---'''
    batch_size = 2
    num_class  = 2
    epochs     = 5

    train_loader = DataLoader(dataset_train,batch_size=batch_size   ,shuffle=True ,pin_memory=True)
    valid_loader = DataLoader(dataset_valid,batch_size=batch_size*10,shuffle=False,pin_memory=True)
    test_loader  = DataLoader(dataset_test ,batch_size=batch_size*10,shuffle=False,pin_memory=True)

    model = Consultformer(batch_size=batch_size,num_class=num_class)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    summary(model)


    for epoch in range(1,epochs+1):
        train(model,train_loader,optimizer,'cpu',epoch,writer)
        validate(model,valid_loader,'cpu',epoch,writer)

    test(model,test_loader,'cpu',1,writer)
    




if __name__ == '__main__':
    main()