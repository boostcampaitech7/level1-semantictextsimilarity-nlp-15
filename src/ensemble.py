import argparse
import glob
import os
import torch
import time

import pandas as pd
import yaml

def bagging_ensemble(args):
    # Get Model List from config.yaml
    with open(os.path.join('config.yaml')) as f:
        configs = yaml.safe_load(f)

    model_list = [configs['model'][key] for key in configs['model'] if 'model_name' in key]

    # Set File Pattern
    name = model_list[-1].replace("/", "-")
    file_pattern = f'{args.output_path}/{name}*_*_*.csv'

    # Read CSV files containing matching patterns
    csvs = []
    try:
        matching_files = glob.glob(file_pattern)
        if not matching_files:
            print(f"Warning: No matching file found for {file_pattern}. Skipping this model.")

        csvs = [pd.read_csv(matching_file).iloc[:, -1] for matching_file in matching_files]

    except Exception as e:
        print(f"Error reading file: {e}. Skipping this model.")

    if not csvs:
        print("No CSV files were found. Exiting.")
        return

    # Weight-based Ensemble
    weights = torch.Tensor(configs['model']['ensemble_weight'])  # weights를 Tensor로 변환
    predictions = torch.stack([torch.Tensor(csv.values) for csv in csvs], dim=1)

    ensemble_predictions = predictions * weights
    ensemble_predictions = ensemble_predictions.sum(dim=1)
    ensemble_predictions = torch.clamp(ensemble_predictions, min=0, max=5)  # 예측값 범위 조정

    result = list(round(float(elem), 1) for elem in ensemble_predictions)

    # Save Ensemble Result
    ensemble_name = name
    submission = pd.read_csv(os.path.join(args.output_path, "sample_submission.csv"))
    submission['target'] = result
    submission.to_csv(os.path.join(args.output_path, f"{ensemble_name}_bagging.csv"), index=False)


def ensemble(args):
    # Get Model List from config.yaml
    with open(os.path.join('config.yaml')) as f:
        configs = yaml.safe_load(f)

    model_list = [configs['model'][key] for key in configs['model'] if 'model_name' in key]

    # Set File Pattern
    name = model_list[-1].replace("/", "-")
    file_pattern = f'{args.output_path}/{name}*_*_*.csv'

    # Read CSV files containing matching patterns
    csvs = []
    try:
        matching_files = glob.glob(file_pattern)
        if not matching_files:
            print(f"Warning: No matching file found for {file_pattern}. Skipping this model.")

        csvs = [pd.read_csv(matching_file).iloc[:, -1] for matching_file in matching_files]

    except Exception as e:
        print(f"Error reading file: {e}. Skipping this model.")

    if not csvs:
        print("No CSV files were found. Exiting.")
        return

    # Ensemble
    weights = torch.Tensor(configs['model']['ensemble_weight'])  # weights를 Tensor로 변환
    predictions = torch.stack([torch.Tensor(csv.values) for csv in csvs], dim=1)
    ensemble_predictions = (predictions * weights).sum(dim=1) if len(weights) == len(predictions) else predictions.mean(dim=1)

    result = list(round(float(elem), 1) for elem in ensemble_predictions)

    # Save Ensemble Result
    # ensemble_name = "_".join([model.replace("/", "-") + f"_{weight:.1f}" for model, weight in zip(model_list, weights)])
    ensemble_name = time.strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(os.path.join(args.output_path, "sample_submission.csv"))
    df['target'] = result
    df.to_csv(os.path.join(args.ensemble_output_path, f"{ensemble_name}.csv"), index=False)

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
