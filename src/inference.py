import argparse
import pandas as pd
import random
import yaml

import pytorch_lightning as pl
import torch

from util import util

def inference(args):
    model_name = args.model_name.replace("/", "-")
    epoch = args.max_epoch

    dataloader = util.Dataloader(
        args.model_name, args.batch_size, args.max_length, args.num_workers, args.train_path, args.val_path, args.dev_path, args.predict_path
    )

    model = torch.load(f'model/{model_name}_{epoch}.pt')

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=epoch
    )

    predictions = trainer.predict(model=model, datamodule=dataloader)
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    file_name = args.output_path + f"/{model_name}_{epoch}.csv"

    output = pd.read_csv('output/sample_submission.csv')
    output['target'] = predictions
    output.to_csv(file_name, index=False)


if __name__ == "__main__":
    # seed 고정
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)

    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()

    with open('config/config.yaml') as f:
        configs = yaml.safe_load(f)

    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../util/train.csv')
    parser.add_argument('--val_path', default='../util/dev.csv')
    parser.add_argument('--dev_path', default='../util/dev.csv')
    parser.add_argument('--predict_path', default='../util/test.csv')
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--checkpoints', default='../checkpoint')
    parser.add_argument('--num_labels', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args(args=[])

    inference(args)