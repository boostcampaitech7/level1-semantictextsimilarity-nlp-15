import os
import argparse
import os
import random
import sys
import yaml

import pytorch_lightning as pl
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util import util
from src.model import model

def train(args):
    model_name = args.model_name.replace("/", "-")
    epoch = args.max_epoch

    # 모델 파일 저장 전 디렉토리 확인 및 생성
    os.makedirs(os.path.join("team", "src", "model"), exist_ok=True)

    dataloader = util.Dataloader(
        args.model_name, args.batch_size, args.max_length, args.num_workers, args.train_path, args.val_path, args.dev_path, args.predict_path
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_pearson",
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode="max"
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_path,
        monitor='val_pearson',
        filename=f"{model_name}_{{epoch}}_{{val_pearson:.4f}}",  # 파일명에 현재 에폭 수와 val_pearson 추가
        save_top_k=1,
        mode="max"
    )

    model_instance = model.Model(args.model_name, args.num_labels, args.learning_rate)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint],
        num_sanity_val_steps=0,
        max_epochs=epoch
    )

    trainer.fit(model=model_instance, datamodule=dataloader)

    # Get the best val_pearson from the checkpoint
    val_pearson = trainer.callback_metrics["val_pearson"].item()
    current_epoch = trainer.current_epoch  # 현재 에폭 가져오기
    torch.save(model_instance, os.path.join('src', 'model', f'{model_name}_{current_epoch}_{val_pearson:.4f}.ckpt'))

def val_only(args):
    # This Function Only takes validation util and show pearson score on a console
    model_name = args.model_name.replace("/", "-")

    # 모델 파일 저장 전 디렉토리 확인 및 생성
    os.makedirs(os.path.join("team", "src", "model"), exist_ok=True)
        
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
        filename="best-checkpoint",
        monitor="val_pearson",
        save_top_k=1,
        mode="max"  # mode should be "max" to find the best val_pearson
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint],
        num_sanity_val_steps=0,
        max_epochs=args.max_epoch
    )

    # Load the checkpoint to get the best val_pearson
    checkpoint_path = pl.callbacks.ModelCheckpoint(dirpath=args.checkpoint_path, filename="best-checkpoint").best_model_path
    checkpoint = torch.load(checkpoint_path)

    # val_pearson을 체크포인트에서 추출
    val_pearson = checkpoint['callbacks'][0]['val_pearson']
    # 현재 에폭 수를 체크포인트에서 직접 가져오는 방식으로 변경
    current_epoch = int(checkpoint_path.split('_')[1])  # filename에서 epoch 추출

    model = torch.load(os.path.join('src', 'model', f'{model_name}_{current_epoch}_{val_pearson:.4f}.ckpt'))
    trainer.test(model=model, datamodule=dataloader)

if __name__ == "__main__":
    # seed 고정
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)

    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed')
    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='data/train.csv')
    parser.add_argument('--val_path', default='data/dev.csv')
    parser.add_argument('--dev_path', default='data/dev.csv')
    parser.add_argument('--predict_path', default='data/test.csv')
    parser.add_argument('--checkpoint_path', default='checkpoint', type=str)
    parser.add_argument('--num_labels', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args(args=[])

    with open(os.path.join('src', 'config', 'config.yaml')) as f:
        configs = yaml.safe_load(f)

    model_list = list(configs['model'].values())

    for m in model_list:
        print("***** Pearson Score Start *****\nModel Name : ", m)
        args.model_name = m
        train(args)
        # val_only(args)
