'''
This file needs to train a custom model defined in functions/model.py,
notebook starting to be tedious
'''

from functions.data import EHR
from functions.runner import train, test
from functions.h_transformer1d_v2 import HTransformer1D_custom

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime as dt

from tokenizers import ByteLevelBPETokenizer

import os
import random
import argparse


def main(epochs: int = 10, batch_size: int = 32, learning_rate: float = 3e-5, weight_decay: float = 1e-4):
    # Imposta il seed per Python, NumPy e PyTorch
    seed = 42
    
    # Imposta il seed per Python
    random.seed(seed)
    
    log_dir = f'./sandboxes/07_consults/David_Transformer/results/{dt.now().strftime("%Y%m%d%H%M")}'
    
    # Verifica se la cartella esiste
    if os.path.exists(log_dir):
        os.rmdir(log_dir) 
    
    writer = SummaryWriter(log_dir = log_dir)
    
    os.makedirs(os.path.join(writer.log_dir,'checkpoint'))
    
    root = 'T:\\lab_research\\RES-Folder-UPOD\\ODIN-UC4\\G_Output\\2_Data\\data\\whole_text_dataset'
    
    # Viene caricato il dataset di train
    dataset_train = EHR(root,'train')
    
    # Viene caricato il dataset di test
    dataset_test  = EHR(root,'test' )
    
    # Definizione degli iperparametri
    hyperparameters = {
        'epochs'        : epochs,
        'num_classes'   : 2,
        'batch_size'    : batch_size,
        'learning_rate' : learning_rate,
        'weight_decay'  : weight_decay,
        'amsgrad'       : True
    }
    
    # Si definisce il DataLoader per il training set
    train_loader = DataLoader(
        dataset_train,
        batch_size = hyperparameters['batch_size'],
        shuffle = True,
        pin_memory = True
    )
    
    # train_collator 
    # https://huggingface.co/docs/transformers/main_classes/data_collator

    #valid_loader = DataLoader(dataset_valid,batch_size=batch_size*10,shuffle=False,pin_memory=True)
    
    # Dato che la fase di test è più rapida poiché richiede solamente inferenza,
    # si utilizza un batch_size più grande
    test_loader  = DataLoader(
        dataset_test,
        batch_size = hyperparameters['batch_size']*5,
        shuffle = False,
        pin_memory = True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer_vocab_path  = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/vocab.json'
    tokenizer_merges_path = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/merges.txt' 
    
    # Inizializzazione del tokenizer che effettua una
    # tokenizzazione basata su Byte Pair Encoding (BPE),
    # algoritmo di tokenizzazione che unisce coppie di byte o 
    # caratteri più frequenti in sequenze. In questo modo è 
    # in grado di creare nuovi tokens basati sulla frequenza delle
    # coppie di caratteri nei dati di addestramento. Questo approccio
    # aiuta a gestire parole rare o sconosciute creando token basati
    # su parti comuni di parole
    tokenizer = ByteLevelBPETokenizer(
        vocab = tokenizer_vocab_path,
        merges = tokenizer_merges_path,
        add_prefix_space = True,
    )
    
    # Definizione del modello utilizzato
    model = HTransformer1D_custom(
        tokenizer = tokenizer,
        device = device,
        num_classes = hyperparameters['num_classes']
    )
    
    # Scelta dell'ottimizzatore (potrebbe essere anche il gradiente discendente)
    optimizer = optim.Adam(
        model.parameters(), 
        lr = hyperparameters['learning_rate'],
        weight_decay = hyperparameters['weight_decay'],
        amsgrad = hyperparameters['amsgrad']
    )

    # Scelta della loss function
    criterion = nn.CrossEntropyLoss()

    print(model)

    # Addestramento e valutazione delle perfomance del modello
    for epoch in range(1, hyperparameters['epochs'] + 1):
        
        print("Starting training epoch")
        
        train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            writer
        )
        
        #print("Ended training epoch")
        
        test(
            model,
            test_loader,
            device,
            epoch,
            writer
        )
        
        #print("Endend test epoch")
    
    
    
    # Non mi ricordo cosa fa questo riga di codice
    test(
        model,
        test_loader,
        device,
        epoch+1,
        writer
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train a custom model, set model arguments")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")

    args = parser.parse_args()

    main(**args.__dict__)