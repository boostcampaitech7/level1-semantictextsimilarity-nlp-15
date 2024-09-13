import pytorch_lightning as pl
import torch

from typing import Any
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoModel, ElectraModel, ElectraTokenizer, get_cosine_schedule_with_warmup

class Model(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.lr = lr
        self.loss_function = nn.L1Loss()
        self.predictions = []

    def forward(self, **x):
        x = self.model(**x) #  SequenceClassifierOutput
        return x.logits.squeeze(-1)

    def batch_to_loss(self, batch: Any):
        input = {key : value for key, value in batch.items() if key != "labels"}
        labels = batch["labels"]
        pred = self(**input)
        loss = self.loss_function(pred, labels.float())
        pearson = torchmetrics.functional.pearson_corrcoef(pred, labels)
        return loss, pearson

    def training_step(self, batch, batch_idx):
        loss, pearson = self.batch_to_loss(batch)
        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pearson = self.batch_to_loss(batch)
        self.log("val_loss", loss, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input = {key: value for key, value in batch.items() if key != "label"}
        labels = batch["labels"]
        pred = self(**input)
        self.predictions.append(pred.detach().cpu())
        pearson = torchmetrics.functional.pearson_corrcoef(pred, labels.to(torch.float64))
        self.log("test_pearson", pearson, logger=True)

    def predict_step(self, batch, batch_idx):
        input = {key: value for key, value in batch.items() if key != "label"}
        pred = self(**input)
        return pred.squeeze(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = {
            'scheduler': get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=20
            ),
            'interval': 'step',
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def setup(self, stage='fit'):
        self.predictions = []