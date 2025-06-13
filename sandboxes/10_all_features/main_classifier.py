from functions.dataset import EHR
from functions.runner import train, test, validate
from functions.model_htransformer import HTransformer1D_custom_averagePooling, \
    HTransformer1D_custom_clsPooling
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

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

from imblearn.over_sampling import RandomOverSampler


def main(
        epochs                  : int   = 10, 
        model_name              : str   = 'Resnet',
        batch_size              : int   = 32,
        learning_rate           : float = 3e-5, 
        weight_decay            : float = 1e-4,
        tokens_len              : int   = 8192,
        embedding_dim           : int   = 512,
        num_classes             : int   = 2,
        amsgrad                 : bool  = True,
        n_splits                : int   = 5,
        test_size               : float = 0.2,
        desired_percentage      : float = 0.3,
        results_folder          : str   = r'results',
        pooling                 : str   = 'cls',
        dataset_path            : str   = r'T:\lab_research\RES-Folder-UPOD\ODIN-UC4\G_Output\2_Data\parquet\df_consults_medication.parquet',
        seed                    : int   = 42,
        tokenizer_path : str = r'D:\LanguageModels\Models\Tokenizers\BPE_tokenizer_medroberta'
    ):
    
    #set_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Imposta questa directory come directory corrente
    os.chdir(script_dir)
    
    tokenizer_vocab_path  = os.path.join(tokenizer_path, 'vocab.json')
    tokenizer_merges_path = os.path.join(tokenizer_path, 'merges.txt') 
    
    
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
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    scaler = StandardScaler()
    
    num_tokens = tokenizer.vocab_size
    
    #print(num_tokens)
    
    df = pd.read_parquet(dataset_path)
    
    labels = df.iloc[:, -1]
    df = df.iloc[:, 1:-1]
    class_counts = labels.value_counts()
    
    majority_class = class_counts.max()
    
    desired_minority_samples = int(desired_percentage * len(labels))
    
    alpha = desired_minority_samples / majority_class
    
    if desired_percentage:
        oversampler = RandomOverSampler(sampling_strategy = alpha,
            random_state=seed)
        
    else:
        oversampler = None
    
    tabular_features = df.select_dtypes(include=['number']).columns.tolist()
    unstructured_features = df.select_dtypes(include=['object']).columns.tolist()
    
    indices = np.arange(df.shape[0])
    
    trainval_indices, test_indices = train_test_split(
        indices,
        test_size=test_size, 
        stratify=labels, 
        random_state=seed
    )
    
    df_test = df.iloc[test_indices]
    labels_test = labels.iloc[test_indices]
    
    df_trainval = df.iloc[trainval_indices]
    labels_trainval = labels.iloc[trainval_indices]
    
    #print(f"Tokenizer's vocab size: {num_tokens}\n")
    
    # Definizione del modello utilizzato
    if model_name == 'HTransformer':
        if pooling == 'avg':
            log_dir = f'HTransformer1D\\{dt.now().strftime("%Y_%m_%d_%H_%M")}_AVG'
        
        else:
            log_dir = f'HTransformer1D\\{dt.now().strftime("%Y_%m_%d_%H_%M")}_CLS'
    
    elif model_name == 'Resnet':        
        log_dir = f'ResConsultNet\\{dt.now().strftime("%Y_%m_%d_%H_%M")}'
    
    # Scelta della loss function
    criterion = nn.CrossEntropyLoss()
    
    best_acc   = -float('inf')
    best_f1    = -float('inf')
    best_prc   = -float('inf')
    best_rec   = -float('inf')
    best_mcc   = -float('inf')
    best_model = None
    
    skf = StratifiedKFold(
        n_splits = n_splits,
        shuffle=True,
        random_state=seed
    )
    
    results_path = os.path.join(script_dir, results_folder)
    os.makedirs(results_path, exist_ok=True)
    
    model_path = os.path.join(results_path, log_dir)
    os.makedirs(model_path, exist_ok=True)
    
    '''
    print(script_dir, '\n')
    print(results_path, '\n')
    print(model_path, '\n')
    print(txt_path, '\n')
    '''
    
    test_performance = []
    
    # Start of the k-fold cross-validation
    for split, (train_indices, val_indices) in enumerate(skf.split(df_trainval, labels_trainval)):
        if model_name == 'HTransformer':
            if pooling == 'avg':
                model = HTransformer1D_custom_averagePooling(
                    depth=3,
                    dim_head=32,
                    ff_mult=4,
                    heads=4,
                    num_classes = num_classes,
                    max_seq_len = tokens_len,
                    block_size = 32,
                    embedding_dim = embedding_dim,
                    num_tokens = num_tokens
                )
            
            else:
                model = HTransformer1D_custom_clsPooling(
                    depth=6,
                    dim_head=32,
                    ff_mult=4,
                    heads=8,
                    num_classes = num_classes,
                    max_seq_len = tokens_len,
                    block_size = 32,
                    embedding_dim = embedding_dim,
                    num_tokens = num_tokens
                )
        
        elif model_name == 'Resnet':
            model = ResConsultNet(
                vocab_size = num_tokens,
                resnet_layers = [2, 2, 2, 2],
                pool_size = 4,
                max_seq_len = tokens_len,
                num_classes = num_classes,
                base_filters = 16
            )
        
        model.to(device)
        
        # Scelta dell'ottimizzatore (potrebbe essere anche il gradiente discendente)
        optimizer = optim.Adam(
            model.parameters(), 
            lr = learning_rate,
            weight_decay = weight_decay,
            amsgrad = amsgrad
        )
        
        # Creazione del writer per ogni split
        writer_path = os.path.join(model_path, f'split_{split}')
        writer = SummaryWriter(log_dir = writer_path)
        
        df_train = df_trainval.iloc[train_indices]
        labels_train = labels_trainval.iloc[train_indices]
        
        if oversampler:
            df_train, labels_train = oversampler.fit_resample(df_train, labels_train)
        
        '''
        # Conta le occorrenze di ciascuna classe
        class_counts = labels_train.value_counts()
        
        # Calcola la percentuale di ciascuna classe
        class_percentages = (class_counts / len(labels_train)) * 100
        
        print(f"Percentuali delle classi:\n {class_percentages}")
        '''
        
        df_val = df_trainval.iloc[val_indices]
        labels_val = labels_trainval.iloc[val_indices]
        
        
        # Viene caricato il dataset di training
        dataset_train = EHR(
            df = df_train,
            labels = labels_train,
            tokens_len = tokens_len,
            tabular_features = tabular_features,
            scaler = scaler,
            unstructured_features = unstructured_features,
            tokenizer = tokenizer
        )
        
        # Viene caricato il dataset di validation
        dataset_val = EHR(
            df = df_val,
            labels = labels_val,
            tokens_len = tokens_len,
            tabular_features = tabular_features,
            scaler = scaler,
            unstructured_features = unstructured_features,
            tokenizer = tokenizer
        )
        
        # Si definisce il DataLoader per il training set
        dataloader_train = DataLoader(
            dataset_train,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = True
        )
        
        # Si definisce il DataLoader per il validation set
        dataloader_val = DataLoader(
            dataset_val,
            batch_size = batch_size*2,
            shuffle = False,
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
                dataloader = dataloader_train,
                optimizer = optimizer,
                criterion = criterion,
                device = device,
                epoch = epoch,
                writer = writer
            )
            
            model, model_acc, model_f1, model_prc, model_mcc, model_rec = validate(
                model = model,
                dataloader = dataloader_val,
                device = device,
                epoch = epoch,
                criterion = criterion,
                writer = writer
            )
            
            # Salvo il modello migliore di uno split
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
                'split'     : split,
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
        
        # Viene caricato il dataset di test
        dataset_test = EHR(
            df = df_test,
            labels = labels_test,
            tabular_features = tabular_features,
            scaler = scaler,
            unstructured_features = unstructured_features,
            tokenizer = tokenizer,
            tokens_len = tokens_len
        )
        
        # Dato che la fase di test è più rapida poiché richiede solamente inferenza,
        # si utilizza un batch_size più grande
        dataloader_test  = DataLoader(
            dataset_test,
            batch_size = batch_size*2,
            shuffle = False,
            pin_memory = True
        )
        
        # Valuto le performance del miglior classificatore di uno split
        # utilizzando il test set
        model, model_acc, model_f1, model_prc, model_mcc, model_rec = test(
            model = split_best_model,
            dataloader = dataloader_test,
            device = device,
            epoch = epoch,
            writer = writer
        )
        
        test_performance.append([model_acc*100, model_f1*100, model_prc*100, model_mcc, model_rec*100])
        
        # Salvo i pesi del modello migliore tra tutti gli split
        if model_mcc > best_mcc:
            best_model  = model
            best_acc    = model_acc
            best_f1     = model_f1
            best_prc    = model_prc
            best_rec    = model_rec
            best_mcc    = model_mcc
    
    joblib.dump({
            'model' : best_model,
            
            'accuracy'      : best_acc,
            'f1'            : best_f1,
            'precision'     : best_prc,
            'recall'        : best_rec,
            'mcc'           : best_mcc
        },
        os.path.join(model_path, 'best_model.joblib')
    )
    
    writer.close()
    
    test_performance = np.array(test_performance)
    
    test_performance_avg = test_performance.mean(axis=0)
    test_performance_std = np.std(test_performance, axis=0)
    
    # Salvo le performance del modello
    performance_path = os.path.join(model_path, 'performance_parameters.txt')
    with open(performance_path, 'w') as file:
        file.write(f"Average Test Metrics\n")
        file.write(f"- Accuracy: {test_performance_avg[0]:.2f} ({test_performance_std[0]:.2f})\n")
        file.write(f"- F1: {test_performance_avg[1]:.2f} ({test_performance_std[1]:.2f})\n")
        file.write(f"- Precision: {test_performance_avg[2]:.2f} ({test_performance_std[2]:.2f})\n")
        file.write(f"- MCC: {test_performance_avg[3]:.3f} ({test_performance_std[3]:.3f})\n")
        file.write(f"- Recall: {test_performance_avg[4]:.2f} ({test_performance_std[4]:.2f})\n")
        file.write("\nParameters:\n")
        file.write(f"Model name: {type(model).__name__}\n")
        file.write(f"Learning rate: {learning_rate}\n")
        file.write(f"Oversample percentage: {desired_percentage}\n")
        file.write(f"Epochs: {epochs}\n")
        file.write(f"Number of splits: {n_splits}\n")
        file.write(f"Weight decay: {weight_decay}\n")
        file.write(f"Batch size: {batch_size}\n")
        file.write(f"Tabular features: {tabular_features}\n")
        file.write(f"Unstructured features: {unstructured_features}")



def set_seed(seed: int) -> None:
    # Imposto il random seed per Python
    random.seed(seed)
    
    # Imposto il random seed per Numpy
    np.random.seed(seed)
    
    # Imposto il random seed per PyTorch
    torch.manual_seed(seed)
    
    # Uso solamente algoritmi deterministici per PyTorch
    #torch.use_deterministic_algorithms(True)
    
    # Imposto la riproducibilità per CUDA (se disponibile)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train a custom model, set model arguments")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--desired_percentage", type=float, default=0.3, help="Desired percentage for the minority class")
    parser.add_argument("--model_name", type=str, default='HTransformer', help="Name of the model")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for the optimizer")
    parser.add_argument("--pooling", type=str, default='avg', choices=['avg', 'cls'])
    parser.add_argument("--tokenizer_path", type=str, required=True)
    
    args = parser.parse_args()
    
    main(**args.__dict__)


#? Perché non utilizzare un pre-trained transformer giá presente in commercio?
#TODO: provare un transformer pre-addestrato in olandese