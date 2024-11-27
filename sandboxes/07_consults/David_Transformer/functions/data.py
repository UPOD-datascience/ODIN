from torch.utils.data import Dataset
import torch
import os
import numpy as np
import joblib
import pandas as pd

class EHR(Dataset):
    
    def __init__(
            self,
            root    : str = './text_data',
            split   : str = ''
        ):
        
        super().__init__()
        
        self.all_path = []
        
        # creo una lista di tutti i percorsi ai file parquet in base
        # allo split (train, test) così non riempo la RAM
        for file in os.listdir(os.path.join(root, split)):
            self.all_path.append(os.path.join(root, split, file))
    
    def __len__(self):
        return len(self.all_path)
    
    # Definisco la logica con cui prelevo i dati
    def __getitem__(self, index):
        # 'subject' = dataframe con due colonne:
        # 1 - testo
        # 2 - label
        subject = pd.read_parquet(self.all_path[index])
        
        return subject.text.str.cat(), torch.from_numpy(subject.label.values).long()


def to_padded_inputs(
        subj, 
        ids, 
        label, 
        input_size,set_type : str = 'train'
    ):
    '''
    Parameters
    ---
    - subj: subject id
    - row: vocab ids torch.Tensor
    - set_type: split can be ['train','valid','test']

    '''
    os.makedirs(f'{set_type}/{subj}',exist_ok=True)
    n = ids.shape[1]//input_size # should be 510 + cls + text_section = 512
    if n >= 1:
        for i in range(n): #foreach consult chunk saving chunk and label tuple
            joblib.dump( 
                (torch.concat(
                    [torch.tensor([[i/n]]), # 0 (text start) ... 1 (text end)
                     ids[:,i*input_size:i*input_size+input_size],
                    ],dim = 1),
                label),   
                f'{set_type}/{subj}/chunk_{i}.joblib'
            )
            

        pad_size = input_size-ids[:,i*input_size+input_size:].shape[1]
        #chunks.append(
        joblib.dump( 
            (torch.concat(
                            [
                                torch.tensor([[(i+1)/n]]),
                                ids[:,i*input_size+input_size:],
                                torch.zeros((1,pad_size))
                            ],
                            dim = 1
                        ),
                label), 
                f'{set_type}/{subj}/chunk_{i+1}.joblib'
            )
    
    elif n == 0:
        pad_size = input_size-ids.shape[1]
        i = 0
        joblib.dump( 
            (torch.concat(
                            [
                                torch.tensor([[i]]),
                                ids,
                                torch.zeros((1,pad_size))
                            ],
                            dim = 1
                        ),
                label), 
                f'{set_type}/{subj}/chunk_{i}.pt'
            )


def generate_dataset(
        df          : pd.DataFrame,
        path        : str = '',
        test_split  : float = 0.2,
        random_seed : int = 42,
        valid_split : float = 0.2,
    ):
    '''
    This was the version 1 of the function, oriented to chunked text
    Parameters
    ---
    - df: dataframe with two columns: patient_id (int) and text_ids (torch.Tensor) make sure to reset index
    - test_split: percentage of test
    - valid_split: validation set percentage on the train set
    - path: path/to/root/containing train, valid, test folders
    '''

    rng = np.random.default_rng(seed = random_seed)
    test_split  = test_split
    valid_split = valid_split
    train_split = 1 - test_split

    train_df = df.groupby('label').sample(frac=train_split,random_state=rng) # nice and stratified
    test_df  = df.loc[~df.index.isin(train_df.index)]

    #shuffle_idx = rng.permutation(df.shape[0])

    #train_idx = shuffle_idx[:int(len(shuffle_idx) * train_split)]
    #valid_idx = train_idx  [:int(len(train_idx)   * valid_split)]
    #train_idx = train_idx  [int(len(train_idx)   * valid_split):]
    #test_idx  = shuffle_idx[int(len(shuffle_idx) * train_split):]

    input_size = 510 # + 1 cls_token + 1 text chunk number in [0...1]

    os.chdir(path) # Local hard drive is faster than remore hard drive

    for idx in train_df.index:
        to_padded_inputs(df.studyId_0831[idx],df.ids[idx],df.label[idx],input_size,'train')

    #or idx in valid_idx:
    #   to_padded_inputs(df.studyId_0831[idx],df.ids[idx],df.label[idx],input_size,'valid')

    for idx in test_df.index:
        to_padded_inputs(df.studyId_0831[idx],df.ids[idx],df.label[idx],input_size,'test')


def generate_dataset_whole_text(
        df          : pd.DataFrame, 
        path        : str = '',
        test_split  : float = 0.2,
        valid_split : float = 0.2,
        random_seed : int = 42
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


# Codice di test
if __name__ == '__main__':
    os.chdir(r'C:\Users\jvitale\data')
    data = EHR(root='./whole_text_dataset', split='train')

    print(data.__getitem__(0))