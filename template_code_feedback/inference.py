import argparse
import pandas as pd
import random
import yaml
import os
import pytorch_lightning as pl
import torch

from team.src.util import util
from team.src.model import model  # 모델 클래스를 가져옵니다.

def inference(args):
    model_name = args.model_name.replace("/", "-")

    # 데이터 로더 설정
    dataloader = util.Dataloader(
        args.model_name, args.batch_size, args.max_length, args.num_workers, args.train_path, args.val_path, args.dev_path, args.predict_path
    )

    # 모델 파일 저장 전 디렉토리 확인 및 생성
    model_dir = 'team/src/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 가장 좋은 체크포인트 로드
    checkpoint_files = [f for f in os.listdir(args.checkpoint_path) if f.endswith('.ckpt')]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found.")

    # val_pearson이 가장 높은 체크포인트 선택
    latest_checkpoint = max(checkpoint_files, key=lambda x: float(x.split('_')[-1].split('=')[-1].replace('.ckpt', '')))
    checkpoint_path = os.path.join(args.checkpoint_path, latest_checkpoint)

    # 체크포인트에서 모델 상태 딕셔너리 로드
    checkpoint = torch.load(checkpoint_path)

    # 모델 인스턴스 생성
    model_instance = model.Model(args.model_name, args.num_labels, args.learning_rate)
    model_instance.load_state_dict(checkpoint['state_dict'])  # 상태 딕셔너리 로드

    # val_pearson 값 추출
    val_pearson = float(latest_checkpoint.split('_')[-1].split('=')[-1].replace('.ckpt', ''))

    # current_epoch 값 추출
    current_epoch = int(latest_checkpoint.split('_')[1].split('=')[1])  

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=args.max_epoch
    )

    predictions = trainer.predict(model=model_instance, datamodule=dataloader)
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # 출력 디렉토리 확인 및 생성
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = f"{output_dir}/{model_name}_{current_epoch}_{val_pearson:.4f}.csv"  # 파일명 수정

    output = pd.read_csv('/data/ephemeral/home/team/src/output/sample_submission.csv')
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

    with open('/data/ephemeral/home/team/src/config/config.yaml') as f:
        configs = yaml.safe_load(f)

    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='/data/ephemeral/home/team/data/train.csv')
    parser.add_argument('--val_path', default='/data/ephemeral/home/team/data/dev.csv')
    parser.add_argument('--dev_path', default='/data/ephemeral/home/team/data/dev.csv')
    parser.add_argument('--predict_path', default='/data/ephemeral/home/team/data/test.csv')
    parser.add_argument('--output_path', default='/data/ephemeral/home/team/src/output')
    parser.add_argument('--checkpoint_path', default='/data/ephemeral/home/team/checkpoint')
    parser.add_argument('--num_labels', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args(args=[])

    inference(args)
