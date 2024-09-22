import torch.utils.data
import pytorch_lightning as pl
import pandas as pd

from . import data_augmentation
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

class Tokenizer(Dataset):
    def __init__(self, model_name, aug_list, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
        self.aug_list = aug_list

    def __len__(self):
        return len(self.tokenizer)

    def __getitem__(self, idx):
        return self.tokenizer[idx]


class Dataset(Dataset):
    def __init__(self, tokenizer, aug_list, path, predict=False):
        self.data = None
        
        self.data = pd.read_csv(path)
        self.aug_list = aug_list
        self.tokenizer = tokenizer
        self.predict = predict

        self.aug_for_dataset_class()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s1 = self.data.iloc[idx]['sentence_1']
        s2 = self.data.iloc[idx]['sentence_2']

        enc = self.tokenizer(s1, s2, return_tensors='pt', truncation=True)

        mapper = {key : value.squeeze(0) for key, value in enc.items()} # value Vector (1, token_size) -> (token_size)

        if not self.predict:
            mapper['labels'] = torch.tensor(self.data.iloc[idx]['label'])

        return mapper

    def aug_for_dataset_class(self):
        if 'swap' in self.aug_list:
            self.data = data_augmentation.swap_sentences(self.data)

        if 'koeda' in self.aug_list:
            self.data = data_augmentation.koeda(self.data)

        if 'remove_special' in self.aug_list:
            self.data = data_augmentation.remove_special_characters(self.data)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, aug_list, batch_size, max_length, num_worker, train_path, val_path, dev_path, predict_path):
        super().__init__()
        self.aug_list = aug_list
        self.tokenizer = Tokenizer(model_name, aug_list, max_length)
        self.batch_size = batch_size
        self.num_worker = num_worker

        self.train_path = train_path
        self.val_path = val_path
        self.dev_path = dev_path
        self.predict_path = predict_path

        self.collate = DataCollatorWithPadding(tokenizer=self.tokenizer.tokenizer)

        self.train_dataset = Dataset(self.tokenizer.tokenizer, aug_list, self.train_path)
        self.val_dataset = Dataset(self.tokenizer.tokenizer, aug_list, self.val_path)
        self.dev_dataset = Dataset(self.tokenizer.tokenizer, aug_list, self.dev_path)
        self.predict_dataset = Dataset(self.tokenizer.tokenizer, aug_list, self.predict_path, predict=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, persistent_workers=True, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker, collate_fn=self.collate)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, persistent_workers=True, batch_size=self.batch_size, num_workers=self.num_worker, collate_fn=self.collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset, persistent_workers=True, batch_size=self.batch_size, num_workers=self.num_worker, collate_fn=self.collate)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, persistent_workers=True, batch_size=self.batch_size, num_workers=self.num_worker, collate_fn=self.collate)