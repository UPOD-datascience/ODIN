import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

import pandas as pd


class EHR(Dataset):
    def __init__(
            self,
            df                      : pd.DataFrame,
            labels                  : pd.Series,
            tabular_features        : list,
            unstructured_features   : list,
            tokenizer               : object,
            tokens_len              : int,
            scaler                  : StandardScaler
        ):
        
        # assert if there are any tabular features, that they are all numerical, otherwise raise error
        for feature in tabular_features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                raise ValueError(f"The column '{feature}' doesn't have numerical data.")
        
        dataset_tabular = df[tabular_features].astype('float32')
        dataset_unstructured = df[unstructured_features]
        
        if not hasattr(scaler, 'scale_'):
            #print('ENTRATO')
            dataset_tabular[tabular_features] = scaler.fit_transform(dataset_tabular)
        else:
            dataset_tabular[tabular_features] = scaler.transform(dataset_tabular)
        
        for feature in unstructured_features:
            tokens = tokenizer(
                list(dataset_unstructured[feature]),
                truncation = True, 
                padding = 'max_length', 
                max_length = tokens_len,
                return_tensors = 'np'
            )
            
            #print(tokens['input_ids'].tolist())
            
            dataset_unstructured.loc[:, feature] = tokens['input_ids'].tolist()
        
        self.dataset_tabular = dataset_tabular
        self.dataset_unstructured = dataset_unstructured
        self.labels = labels
        self.masks = tokens['attention_mask'].tolist()
    
    def __getitem__(self, index):
        return torch.tensor(self.dataset_tabular.iloc[[index]].values[0][:], dtype=torch.float), \
            torch.tensor(self.dataset_unstructured.iloc[[index]].values[0][0], dtype=torch.long), \
            torch.tensor(self.masks[index], dtype=torch.bool), \
            torch.tensor(self.labels.iloc[index], dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)


# Test main
if __name__ == '__main__':
    
    tokenizer_vocab_path  = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/vocab.json'
    tokenizer_merges_path = './sandboxes/07_consults/pretrained/DutchEHRTokenizer/merges.txt' 
    
    root = r'L:\lab_research\RES-Folder-UPOD\ODIN-UC4\G_Output\2_Data\data\kfold_shuffle\fold_0'
    
    # Viene caricato il dataset di train
    dataset_train = EHR(
        root = root,
        split = 'train'
    )
    
    print(dataset_train.__getitem__(3))

