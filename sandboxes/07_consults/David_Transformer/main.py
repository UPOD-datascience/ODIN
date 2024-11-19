'''
This file needs to train a custom model defined in functions/model.py,
notebook starting to be tedious
'''

from functions.data import EHR
from functions.runner import train, test
from functions.model import ConsultFormer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime as dt

import os


def main():
    
    writer = SummaryWriter(
        log_dir = f'./sandboxes/07_consults/David_Transformer/results_{dt.now().strftime("%Y%m%d%H%M%S")}'
    )
    
    os.makedirs(os.path.join(writer.log_dir,'checkpoint'))
    
    root = 'T:\\lab_research\\RES-Folder-UPOD\\ODIN-UC4\\G_Output\\2_Data\\data\\whole_text_dataset'
    
    # Viene caricato il dataset di train
    dataset_train = EHR(root,'train')
    
    # Viene caricato il dataset di test
    dataset_test  = EHR(root,'test' )
    
    # Definizione degli iperparametri
    hyperparameters = {
        'epochs'        : 100,
        'num_class'     : 2,
        'batch_size'    : 256,
        'learning_rate' : 3e-4
    }
    
    # Si definisce il DataLoader per il training set
    train_loader = DataLoader(
        dataset_train,
        batch_size = hyperparameters['batch_size'],
        shuffle = True,
        pin_memory = True
    )
    
    #valid_loader = DataLoader(dataset_valid,batch_size=batch_size*10,shuffle=False,pin_memory=True)
    
    # Dato che la fase di test è più rapida poiché richiede solamente inferenza,
    # si utilizza un batch_size più grande
    test_loader  = DataLoader(
        dataset_test,
        batch_size = hyperparameters['batch_size']*10,
        shuffle=False,
        pin_memory=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Definizione del modello utilizzato
    model = ConsultFormer(
        device = 'cpu'
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
            optimizer,
            criterion,
            device,
            epoch,
            writer
        )
        
        test(
            model,
            test_loader,
            device,
            epoch,
            writer
        )
    
    
    # Non mi ricordo cosa fa questo riga di codice
    test(
        model,
        test_loader,
        device,
        epoch+1,
        writer
    )




if __name__ == '__main__':
    main()
