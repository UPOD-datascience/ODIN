from torch.utils.data import Dataset,DataLoader
import torch
import os
import numpy as np
import joblib


class EHR(Dataset):
    def __init__(self, root : str = './text_data', split : str = ''):
        super().__init__()

        self.all_path = []

        for subj in os.listdir(os.path.join(root,split)):
            for chunk in os.listdir(os.path.join(root,split,subj)):
                self.all_path.append(os.path.join(root,split,subj,chunk))

    def __len__(self):
        return len(self.all_path)

    def __getitem__(self, index):
        ids, label = joblib.load(self.all_path[index])
        return ids.to(int), label


def to_padded_inputs(subj, ids, label, input_size,set_type : str = 'train'):
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

def generate_dataset(df, random_seed : int = 42,test_split : float = 0.2, valid_split : float = 0.2,path:str = ''):
    '''
    Parameters
    ---
    - df: dataframe with two columns: patient_id (int) and text_ids (torch.Tensor) make sure to reset index
    - test_split: percentage of test
    - valid_split: validation set percentage on the train set
    - path: path/to/root/containing train, valid, test folders
    '''

    rng = np.random.default_rng(seed=random_seed)

    shuffle_idx = rng.permutation(df.shape[0])
    test_split  = test_split
    valid_split = valid_split
    train_split = 1 - test_split

    train_idx = shuffle_idx[:int(len(shuffle_idx) * train_split)]
    valid_idx = train_idx  [:int(len(train_idx)   * valid_split)]
    train_idx = train_idx  [int(len(train_idx)   * valid_split):]
    test_idx  = shuffle_idx[int(len(shuffle_idx) * train_split):]

    input_size = 510 # + 1 cls_token + 1 text chunk number in [0...1]

    os.chdir(path) # Local hard drive is faster than remore hard drive

    for idx in train_idx:
        to_padded_inputs(df.studyId_0831[idx],df.ids[idx],df.label[idx],input_size,'train')

    for idx in valid_idx:
        to_padded_inputs(df.studyId_0831[idx],df.ids[idx],df.label[idx],input_size,'valid')

    for idx in test_idx:
        to_padded_inputs(df.studyId_0831[idx],df.ids[idx],df.label[idx],input_size,'test')