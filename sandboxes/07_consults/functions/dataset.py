import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from tokenizers import ByteLevelBPETokenizer

from transformers import PreTrainedTokenizerFast

import os

import numpy as np

from sklearn.model_selection import StratifiedKFold


class EHR(Dataset):
    def __init__(
            self,
            root: str, 
            split: str
        ):
        
        self.all_path = []
        
        # Supponiamo che tu abbia una struttura di file simile
        for file in os.listdir(os.path.join(root, split)):
            self.all_path.append(os.path.join(root, split, file))
    
    def __len__(self):
        return len(self.all_path)
    
    def __getitem__(self, index):
        # Carica il file Parquet
        subject = pd.read_parquet(self.all_path[index])
        
        # Concatenazione del testo
        text = subject.text.str.cat(sep=" ")
        
        # Restituisci il testo e la label
        return text, torch.tensor(subject.label.values, dtype=torch.long)



# Helpful functions
def collate_fn(
        batch, 
        tokenizer, 
        max_seq_len
    ):
    
    # Separo il testo e le label dal batch
    texts, labels = zip(*batch)
    
    #print(texts)
    
    # Tokenizzo tutti i testi nel batch
    encoding = tokenizer(
        list(texts), 
        truncation = True, 
        padding = 'max_length',
        max_length = max_seq_len
    )
    
    # Restituisco il dizionario di encoding e le label come tensori
    return torch.tensor(encoding['input_ids']), torch.stack(labels)

def generate_consults_dataset(
        df          : pd.DataFrame, 
        random_seed : int = 42,
        test_split  : float = 0.2,
        valid_split : float = 0.2,
        path        : str = ''
    ):
    
    '''
    This is new version for whole text dataset.
    Parameters
    ---
    - df: dataframe with two columns: text (str) and labels make sure to reset index
    - test_split: percentage of test
    - valid_split: validation set percentage on the train set
    - path: path/to/root/containing train, valid, test folders
    '''
    
    rng = np.random.default_rng(seed=random_seed)
    test_split  = test_split
    valid_split = valid_split
    train_split = 1 - test_split
    
    train_df = df.groupby('label').sample(frac=train_split,random_state=rng) # nice and stratified
    test_df  = df.loc[~df.index.isin(train_df.index)]
    
    os.chdir(path) # Local hard drive is faster than remore hard drive
    for idx in train_df.index:
        train_df.query(f'studyId_0831 == {idx}').to_parquet(f'train/{idx}.parquet')    
    
    #for idx in valid_idx:
    #   to_padded_inputs(df.studyId_0831[idx],df.ids[idx],df.label[idx],input_size,'valid')
    
    for idx in test_df.index:
        test_df.query(f'studyId_0831 == {idx}').to_parquet(f'test/{idx}.parquet')   

def generate_merge_dataset(
        df          : pd.DataFrame, 
        random_seed : int = 42,
        test_split  : float = 0.2,
        valid_split : float = 0.2,
        folds       : int = 5,
        path        : str = ''
    ):
    
    '''
    This is new version for whole text dataset.
    Parameters
    ---
    - df: dataframe with two columns: text (str) and labels make sure to reset index
    - test_split: percentage of test
    - valid_split: validation set percentage on the train set
    - folds: number of folds for the k-fold cross-validation
    - path: path/to/root/containing train, valid, test folders
    '''
    
    rng = np.random.default_rng(seed = random_seed)
    
    path = os.path.join(path, f"kfold_{str(random_seed)}")
    
    os.makedirs(path, exist_ok=True)
    
    skf = StratifiedKFold(
        n_splits = folds,
        shuffle = True,
        random_state = random_seed
    )
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df['label'])):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        fold_path = os.path.join(path, f'fold_{fold_idx}')
        os.makedirs(fold_path, exist_ok=True)
        
        if valid_split > 0:
            train_split = 1 - valid_split
            
            # Stratified sample for train split
            train_df = df.groupby('label').sample(
                frac = train_split,
                random_state = rng
            ) 
            
            # Remaining samples for validation
            valid_df = train_df.loc[~train_df.index.isin(train_df.index)]
        
        else:
            valid_df = pd.DataFrame()
        
        # Save the test set
        os.makedirs(os.path.join(fold_path, 'test'), exist_ok=True)
        for idx in test_df.index:
            test_df.query(f'studyId_0831 == {idx}').to_parquet(os.path.join(fold_path, f'test/{idx}.parquet'))
        
        # Save the train set
        os.makedirs(os.path.join(fold_path, 'train'), exist_ok=True)
        for idx in train_df.index:
            train_df.query(f'studyId_0831 == {idx}').to_parquet(os.path.join(fold_path, f'train/{idx}.parquet'))
        
        if not valid_df.empty:
            os.makedirs(os.path.join(fold_path, 'val'), exist_ok=True)
            for idx in valid_df.index:
                valid_df.query(f'studyId_0831 == {idx}').to_parquet(os.path.join(fold_path, f'val/{idx}.parquet'))
    
    print('k-fold cross-validation completed successfully')


# Test main
if __name__ == '__main__':
    
    tokenizer_vocab_path  = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/vocab.json'
    tokenizer_merges_path = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/merges.txt' 
    
    root = 'T:\\lab_research\\RES-Folder-UPOD\\ODIN-UC4\\G_Output\\2_Data\\data\\whole_text_dataset'
    
    tokenizer = ByteLevelBPETokenizer(
        vocab = tokenizer_vocab_path,
        merges = tokenizer_merges_path,
        add_prefix_space = True
    )
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer._tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Crea il dataset
    test_dataset = EHR(root=root, split='test', tokenizer=tokenizer)

    # Crea il DataLoader con la funzione collate_fn personalizzata
    test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=lambda batch: collate_fn(batch, tokenizer, 70))

    # Esempio di utilizzo nel ciclo di training
    for batch in test_loader:
        inputs, labels = batch
        print(inputs)  # Input tokenizzati
        print(labels)  # Etichette
        break

