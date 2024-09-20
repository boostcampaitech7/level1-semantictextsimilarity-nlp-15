import torch.utils.data
import pytorch_lightning as pl
import pandas as pd

from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

class Tokenizer(Dataset):
    def __init__(self, model_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)

    def __len__(self):
        return len(self.tokenizer)

    def __getitem__(self, idx):
        return self.tokenizer[idx]


class Dataset(Dataset):
    def __init__(self, tokenizer, path, predict=False):
        self.data = None

        # if aug_list is not None and "swap" in aug_list:
        #     df = pd.read_csv(path)
        #     swap_df = pd.read_csv(path)

        #     swap_df['sentence_1'], swap_df['sentence_2'] = swap_df['sentence_2'], swap_df['sentence_1']

        #     self.data = pd.concat([df, swap_df])

        # else:
        
        self.data = pd.read_csv(path)

        self.tokenizer = tokenizer
        self.predict = predict

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

class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, max_length, num_worker, train_path, val_path, dev_path, predict_path):
        super().__init__()
        self.tokenizer = Tokenizer(model_name, max_length)
        self.batch_size = batch_size
        self.num_worker = num_worker

        self.train_path = train_path
        self.val_path = val_path
        self.dev_path = dev_path
        self.predict_path = predict_path

        self.collate = DataCollatorWithPadding(tokenizer=self.tokenizer.tokenizer)

        self.train_dataset = Dataset(self.tokenizer.tokenizer, self.train_path)
        self.val_dataset = Dataset(self.tokenizer.tokenizer, self.val_path)
        self.dev_dataset = Dataset(self.tokenizer.tokenizer, self.dev_path)
        self.predict_dataset = Dataset(self.tokenizer.tokenizer, self.predict_path, predict=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, persistent_workers=True, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker, collate_fn=self.collate)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, persistent_workers=True, batch_size=self.batch_size, num_workers=self.num_worker, collate_fn=self.collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset, persistent_workers=True, batch_size=self.batch_size, num_workers=self.num_worker, collate_fn=self.collate)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, persistent_workers=True, batch_size=self.batch_size, num_workers=self.num_worker, collate_fn=self.collate)