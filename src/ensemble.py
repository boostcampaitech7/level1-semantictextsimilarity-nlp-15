import argparse
import glob
import os
import torch
import time

import pandas as pd
from scipy.stats import pearsonr  # Pearson 계수를 계산하기 위한 모듈 추가
import yaml

def bagging_ensemble(args):
    # Get Model List from config.yaml
    with open(os.path.join('src', 'config', 'config.yaml')) as f:
        configs = yaml.safe_load(f)

    # 모델 이름만 가져오기
    model_list = [configs['model'][key] for key in configs['model'] if 'model_name' in key]
    
    csvs = []

    # model_name은 문자열이므로 replace()를 사용할 수 있음
    name = model_list[-1].replace("/", "-")
    # 수정된 파일 경로: epoch와 Pearson 값을 포함
    file_pattern = f'{args.output_path}/{name}*_*_*.csv'  # 와일드카드 사용

    # 예측 결과가 담긴 CSV 파일 읽기
    try:
        # 와일드카드 패턴에 맞는 모든 파일을 읽어오기
        matching_files = glob.glob(file_pattern)
        if not matching_files:
            print(f"Warning: No matching file found for {file_pattern}. Skipping this model.")
        
        # 모든 열을 가져옴
        csvs = [pd.read_csv(matching_file).iloc[:, -1] for matching_file in matching_files] 
    except Exception as e:
        print(f"Error reading file: {e}. Skipping this model.")

    if not csvs:
        print("No CSV files were found. Exiting.")
        return

    # Ensemble
    weights = torch.Tensor(configs['model']['ensemble_weight'])  # weights를 Tensor로 변환
    predictions = torch.stack([torch.Tensor(csv.values) for csv in csvs], dim=1)

    # 가중치를 적용하여 앙상블 예측 계산
    ensemble_predictions = predictions * weights
    ensemble_predictions = ensemble_predictions.sum(dim=1)
    ensemble_predictions = torch.clamp(ensemble_predictions, min=0, max=5)  # 예측값 범위 조정

    result = list(round(float(elem), 1) for elem in ensemble_predictions)

    # Save Ensemble Result
    ensemble_name = name
    submission = pd.read_csv(os.path.join(args.output_path, "sample_submission.csv"))
    submission['target'] = result

    # 결과를 CSV로 저장
    submission.to_csv(os.path.join(args.output_path, f"{ensemble_name}_bagging.csv"), index=False)


def ensemble(args):
    # Get Model List from config.yaml
    with open(os.path.join('config.yaml')) as f:
        configs = yaml.safe_load(f)

    # 모델 이름만 가져오기
    model_list = [configs['model'][key] for key in configs['model'] if 'model_name' in key]
    
    csvs = []

    for model_name in model_list:
        # model_name은 문자열이므로 replace()를 사용할 수 있음
        name = model_name.replace("/", "-")
        # 수정된 파일 경로: epoch와 Pearson 값을 포함
        file_pattern = f'{args.output_path}/{name}_*_*.csv'  # 와일드카드 사용

        # 예측 결과가 담긴 CSV 파일 읽기
        try:
            # 와일드카드 패턴에 맞는 모든 파일을 읽어오기
            matching_files = glob.glob(file_pattern)
            if not matching_files:
                print(f"Warning: No matching file found for {file_pattern}. Skipping this model.")
                continue
            
            # 마지막 열을 가져옴
            label = pd.read_csv(matching_files[-1]).iloc[:, -1]  # 가장 최근 파일 선택
            csvs.append(label)
            
        except Exception as e:
            print(f"Error reading file: {e}. Skipping this model.")
            continue

    if not csvs:
        print("No CSV files were found. Exiting.")
        return

    # Ensemble
    weights = torch.Tensor(configs['model']['ensemble_weight'])  # weights를 Tensor로 변환
    predictions = torch.stack([torch.Tensor(csv.values) for csv in csvs], dim=1)

    # 가중치를 적용하여 앙상블 예측 계산
    ensemble_predictions = (predictions * weights).sum(dim=1) if len(weights) == len(predictions) else predictions.mean(dim=1)
    #ensemble_predictions = torch.clamp(ensemble_predictions, min=0, max=5)  # 예측값 범위 조정

    result = list(round(float(elem), 1) for elem in ensemble_predictions)

    # Save Ensemble Result
    # ensemble_name = "_".join([model.replace("/", "-") + f"_{weight:.1f}" for model, weight in zip(model_list, weights)])
    ensemble_name = time.strftime("%Y%m%d_%H%M%S")

    submission = pd.read_csv(os.path.join(args.output_path, "sample_submission.csv"))
    submission['target'] = result

    # 결과를 CSV로 저장
    submission.to_csv(os.path.join(args.ensemble_output_path, f"{ensemble_name}.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--num_labels', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)

    parser.add_argument('--train_path', default='data/train.csv')
    parser.add_argument('--val_path', default='data/dev.csv')
    parser.add_argument('--dev_path', default='data/dev.csv')
    parser.add_argument('--predict_path', default='checkpoint') 
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--checkpoint_path', default='checkpoint')

    args = parser.parse_args(args=[])

    ensemble(args)
