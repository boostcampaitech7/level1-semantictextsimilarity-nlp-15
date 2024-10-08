import pandas as pd
import pytorch_lightning as pl
import torch

import argparse
import random
import yaml
import os
import sys

from util import util
from model import model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inference(args, idx=-1):
    model_name = args.model_name.replace("/", "-")

    # Setup dataloader
    dataloader = util.Dataloader(
        args.model_name, args.aug_list, args.batch_size, args.max_length, args.num_workers, args.train_path, args.val_path, args.dev_path, args.predict_path
    )

    # Load best checkpoint
    checkpoint_files = [f for f in os.listdir(args.checkpoint_path) if f.endswith('.ckpt') and model_name in f]

    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found.")

    latest_checkpoint = max(checkpoint_files, key=lambda x: float(x.split('_')[-1].split('=')[-1].replace('.ckpt', '')))
    checkpoint_path = os.path.join(args.checkpoint_path, latest_checkpoint)

    checkpoint = torch.load(checkpoint_path)

    # Setup model
    model_instance = model.Model(args.model_name, args.num_labels, args.learning_rate)
    model_instance.load_state_dict(checkpoint['state_dict'])

    val_pearson = float(latest_checkpoint.split('_')[-1].split('=')[-1].replace('.ckpt', ''))
    current_epoch = int(latest_checkpoint.split('_')[1].split('=')[1])

    # Setup trainer and predict
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=args.max_epoch
    )

    predictions = trainer.predict(model=model_instance, datamodule=dataloader)
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # Save predictions
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = pd.read_csv(os.path.join(args.output_path, "sample_submission.csv"))
    output['target'] = predictions

    if idx == -1:
        output.to_csv(os.path.join(output_dir, f"{model_name}_{current_epoch}_{val_pearson:.4f}.csv"), index=False)
    else:
        output.to_csv(os.path.join(output_dir, f"{model_name}{idx}_{current_epoch}_{val_pearson:.4f}.csv"), index=False)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)

    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()

    with open(os.path.join("src", "config", "config.yaml")) as f:
        configs = yaml.safe_load(f)

    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='data/train.csv')
    parser.add_argument('--val_path', default='data/dev.csv')
    parser.add_argument('--dev_path', default='data/dev.csv')
    parser.add_argument('--predict_path', default='data/test.csv')
    parser.add_argument('--output_path', default='src/output')
    parser.add_argument('--checkpoint_path', default='checkpoint')
    parser.add_argument('--num_labels', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args(args=[])

    inference(args)