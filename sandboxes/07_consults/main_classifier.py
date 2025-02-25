'''
This file needs to train a custom model defined in 'functions' folder,
notebook starting to be tedious
'''

from functions.dataset import EHR, collate_fn
from functions.runner import train, validate, test
from functions.model_htransformer import HTransformer1D_custom
from functions.model_resnet import ResConsultNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime as dt

from tokenizers import ByteLevelBPETokenizer

from transformers import PreTrainedTokenizerFast

import random

import argparse

import os

import numpy as np

import joblib


def main(
        epochs          : int   = 10, 
        model_name      : str   = 'Resnet',
        batch_size      : int   = 32,
        learning_rate   : float = 3e-5, 
        weight_decay    : float = 1e-4,
        max_seq_len     : int   = 5000,
        embedding_dim   : int   = 256,
        num_classes     : int   = 2,
        amsgrad         : bool  = True,
        n_splits        : int   = 5,
        root            : str   = 'T:\\lab_research\\RES-Folder-UPOD\\ODIN-UC4\\G_Output\\2_Data\\data\\kfold_stratified',
        seed            : int   = 42
    ):
    
    set_seed(seed)
    
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
        add_prefix_space = True
    )
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = tokenizer._tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    num_tokens = tokenizer.vocab_size
    
    #print(f"Tokenizer's vocab size: {num_tokens}\n")
    
    '''
    for inputs, targets in train_loader:
        print(inputs, targets)  # Batch sarà un dizionario con 'input_ids', features, 'labels'
        break
    '''
    
    # Definizione del modello utilizzato
    if model_name == 'HTransformer':        
        log_dir = f'./sandboxes/07_consults/results/HTransformer1D/{dt.now().strftime("%Y_%m_%d_%H_%M")}'
    
    elif model_name == 'Resnet':        
        log_dir = f'./sandboxes/07_consults/results/ResConsultNet/{dt.now().strftime("%Y_%m_%d_%H_%M")}'
    
    # Scelta della loss function
    criterion = nn.CrossEntropyLoss()
    
    best_acc    = -float('inf')
    best_f1     = -float('inf')
    best_prc    = -float('inf')
    best_rec    = -float('inf')
    best_mcc    = -float('inf')
    best_model  = None
    
    # Viene caricato il dataset di test
    dataset_test  = EHR(
        root = root,
        split = 'Test'
    )
    
    # Dato che la fase di test è più rapida poiché richiede solamente inferenza,
    # si utilizza un batch_size più grande
    test_loader  = DataLoader(
        dataset_test,
        batch_size = batch_size*5,
        shuffle = False,
        collate_fn = lambda batch: collate_fn(batch, tokenizer, max_seq_len),
        pin_memory = True
    )
    
    # Start of the k-fold cross-validation
    for split in range(0, n_splits):
        if model_name == 'HTransformer':
            model = HTransformer1D_custom(
                device = device,
                num_classes = num_classes,
                max_seq_len = max_seq_len,
                embedding_dim = embedding_dim,
                num_tokens = num_tokens
            )
        
        elif model_name == 'Resnet':
            model = ResConsultNet(
                vocab_size = num_tokens,
                resnet_layers = [2, 2, 2, 2],
                pool_size = 4,
                device = device,
                num_classes = num_classes,
                base_filters = 16
            )
        
        # Scelta dell'ottimizzatore (potrebbe essere anche il gradiente discendente)
        optimizer = optim.Adam(
            model.parameters(), 
            lr = learning_rate,
            weight_decay = weight_decay,
            amsgrad = amsgrad
        )
        
        # Creazione del path per lo split i-esimo
        split_path = os.path.join(root, f'split_{split}')
        
        # Creazione del writer per ogni split
        writer_dir = os.path.join(log_dir, f'split_{split}')
        writer = SummaryWriter(log_dir = writer_dir)
        
        # Viene caricato il dataset di training
        dataset_train = EHR(
            root = split_path,
            split = 'train'
        )
        
        # Viene caricato il dataset di validation
        dataset_val = EHR(
            root = split_path,
            split = 'val'
        )
        
        # Si definisce il DataLoader per il training set
        train_loader = DataLoader(
            dataset_train,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = lambda batch: collate_fn(batch, tokenizer, max_seq_len),
            pin_memory = True
        )
        
        # Si definisce il DataLoader per il validation set
        val_loader = DataLoader(
            dataset_val,
            batch_size = batch_size*5,
            shuffle = False,
            collate_fn = lambda batch: collate_fn(batch, tokenizer, max_seq_len),
            pin_memory = True
        )
        
        print(f'Split number: {split}\n')
        
        split_best_acc      = -float('inf')
        split_best_f1       = -float('inf')
        split_best_prc      = -float('inf')
        split_best_rec      = -float('inf')
        split_best_mcc      = -float('inf')
        split_best_model    = None
        split_best_epoch    = None
        
        # Addestramento e valutazione delle perfomance del modello
        for epoch in range(1, epochs+1):
            
            #print("Starting training epoch")
            train(
                model = model,
                dataloader = train_loader,
                optimizer = optimizer,
                criterion = criterion,
                device = device,
                epoch = epoch,
                writer = writer
            )
            
            model, model_acc, model_f1, model_prc, model_mcc, model_rec = validate(
                model = model,
                dataloader = val_loader,
                device = device,
                epoch = epoch,
                criterion = criterion,
                writer = writer
            )
            
            # Salva i pesi del modello migliore all'interno dello stesso split
            if model_mcc > split_best_mcc:
                split_best_model = model
                split_best_epoch = epoch
                
                split_best_acc  = model_acc
                split_best_f1   = model_f1
                split_best_prc  = model_prc
                split_best_rec  = model_rec
                split_best_mcc  = model_mcc
        
        joblib.dump({
                'model'     : split_best_model,
                'epoch'     : split_best_epoch,
                
                'accuracy'  : split_best_acc,
                'f1'        : split_best_f1,
                'precision' : split_best_prc,
                'recall'    : split_best_rec,
                'mcc'       : split_best_mcc
            },
            os.path.join(writer.log_dir, f'split_{split}_best_model.joblib')
        )
        
        print(f'split_{split} best epoch: {split_best_epoch}\n')
        
        # Carica i pesi della migliore epoca per il modello
        model = split_best_model
        
        model, model_acc, model_f1, model_prc, model_mcc, model_rec = test(
            model = model,
            dataloader = test_loader,
            device = device,
            epoch = epoch,
            writer = writer
        )
        
        # Salvo i pesi del modello migliore tra tutti gli split
        if model_mcc > best_mcc:
            best_model  = model
            best_acc    = model_acc
            best_f1     = model_f1
            best_prc    = model_prc
            best_rec    = model_rec
            best_mcc    = model_mcc
    
    joblib.dump({
            'model'     : best_model,
            
            'accuracy'  : best_acc,
            'f1'        : best_f1,
            'precision' : best_prc,
            'recall'    : best_rec,
            'mcc'       : best_mcc
        },
        os.path.join(log_dir, 'best_model.joblib')
    )


def set_seed(seed: int) -> None:
    # Imposto il random seed per Python
    random.seed(seed)
    
    # Imposto il random seed per Numpy
    np.random.seed(seed)
    
    # Imposto il random seed per PyTorch
    torch.manual_seed(seed)
    
    # Uso solamente algoritmi deterministici per PyTorch
    torch.use_deterministic_algorithms(True)
    
    # Imposto la riproducibilità per CUDA (se disponibile)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train a custom model, set model arguments")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=600, help="Batch size for training")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--model_name", type=str, default='Resnet', help="Name of the model")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    
    args = parser.parse_args()
    
    main(**args.__dict__)
    
    
#? Perché non utilizzare un pre-trained transformer giá presente in commercio?
#TODO: provare un transformer pre-addestrato in olandese