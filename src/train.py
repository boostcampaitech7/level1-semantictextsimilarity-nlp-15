import argparse
import random
import yaml

import pytorch_lightning as pl
import torch

from util import util

def train(args):
    model_name = args.model_name.replace("/", "-")
    epoch = args.max_epoch

    dataloader = util.Dataloader(
        args.model_name, args.batch_size, args.max_length, args.num_workers, args.train_path, args.val_path, args.dev_path, args.predict_path
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename=f"best_{model_name}_{epoch}",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )

    from model import model

    model = model.Model(args.model_name, args.num_labels, args.learning_rate)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint],
        num_sanity_val_steps=0,
        max_epochs=epoch
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, f'model/{model_name}_{epoch}.pt')

def val_only(args):
    # This Function Only takes validation util and show pearson score on a console

    model_name = args.model_name.replace("/", "-")
    epoch = args.max_epoch

    dataloader = util.Dataloader(
        args.model_name, args.batch_size, args.max_length, args.num_workers, args.train_path, args.val_path, args.dev_path, args.predict_path
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoints,
        filename="best-checkpoint",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint],
        num_sanity_val_steps=0,
        max_epochs=epoch
    )

    model = torch.load(f'model/{model_name}_{epoch}.pt').to('cuda')
    trainer.test(model=model, datamodule=dataloader)


if __name__ == "__main__":
    # seed 고정
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)

    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../util/train.csv')
    parser.add_argument('--val_path', default='../util/dev.csv')
    parser.add_argument('--dev_path', default='../util/dev.csv')
    parser.add_argument('--predict_path', default='../util/test.csv')
    parser.add_argument('--checkpoints', default='../checkpoint')
    parser.add_argument('--num_labels', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args(args=[])

    #train(args)

    with open('config/config.yaml') as f:
        configs = yaml.safe_load(f)

    model_list = list(configs['model'].values())

    for model in model_list:
        print("***** Pearson Score Start *****\nModel Name : ", model)
        args.model_name = model
        val_only(args)