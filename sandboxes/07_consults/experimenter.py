'''
This file needs to train a custom model defined in jacopy/model.py, notebook starting to be tedious
'''

from jacopy.data      import EHR
from jacopy.model     import ResConsultNet
from jacopy.runner    import train, test
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import torch
import os
from joblib import load


def main():
    
    writer = SummaryWriter(
        log_dir = f'./sandboxes/07_consults/experiments/{dt.now().strftime("%Y%m%d%H%M%S")}'
    )
    
    os.makedirs(os.path.join(writer.log_dir,'checkpoint'))
    
    root = 'T:\\lab_research\\RES-Folder-UPOD\\ODIN-UC4\\G_Output\\2_Data\\data\\whole_text_dataset'

    dataset_train = EHR(root,'train')
    dataset_test  = EHR(root,'test' )

    '''--- Hyperparameters ---'''
    hyperparameters = {
        'batch_size'    : 400,
        'num_class'     : 2,
        'epochs'        : 100,
        'learning_rate' : 3e-4
    }

    train_loader = DataLoader(
        dataset_train,
        batch_size = hyperparameters['batch_size'],
        shuffle = True,
        pin_memory = True
    )
    
    #valid_loader = DataLoader(dataset_valid,batch_size=batch_size*10,shuffle=False,pin_memory=True)
    
    # Dato che la fase di test è più rapida poiché richiede solamente inferenza
    # si utilizza un batch_size più grande
    test_loader  = DataLoader(
        dataset_test,
        batch_size = hyperparameters['batch_size']*10,
        shuffle=False,
        pin_memory=True
    )
    
    '''
    for batch in test_loader:
        print(batch)
        break
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ResConsultNet(
        resnet_layers = [2,2,2,2],
        pool_size = 4,
        device = 'cpu',
        base_filters = 16
    )
    
    # Scelta dell'ottimizzatore (potrebbe essere anche il gradiente discendente)
    optimizer = optim.Adam(
        model.parameters(), 
        lr = hyperparameters['learning_rate']
    )
    
    # Scelta della loss function
    criterion = nn.CrossEntropyLoss()
    
    print(model)
    
    # Addestramento e valutazione delle perfomance del modello
    for epoch in range(1, hyperparameters['epochs'] + 1):
        train(
            model,
            train_loader,
            optimizer,criterion,
            device,
            epoch,
            writer
        )
    
    # Non mi ricordo cosa fa questo riga di codice
    test(
        model,
        test_loader,
        device,
        hyperparameters['epochs'] + 1,
        writer
    )




if __name__ == '__main__':
    main()
