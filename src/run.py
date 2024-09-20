import os
import argparse

import torch
import yaml
import random
import warnings
import transformers

from ensemble import ensemble  # ensemble.py에서 ensemble 함수를 가져옵니다.
from train import train
from inference import inference

def set_parser_and_model():
    with open(os.path.join('src', 'config', 'config.yaml')) as f:
        configs = yaml.safe_load(f)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='', type=str)

    parser.add_argument('--seed', default=configs['hyperparameters']['seed'], type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--num_labels', default=configs['hyperparameters']['num_labels'], type=int)
    parser.add_argument('--num_workers', default=configs['hyperparameters']['num_workers'], type=int)

    parser.add_argument('--batch_size', default=configs['hyperparameters']['batch_size'], type=int)
    parser.add_argument('--max_length', default=configs['hyperparameters']['max_length'], type=int)
    parser.add_argument('--max_epoch', default=configs['hyperparameters']['max_epoch'], type=int)
    parser.add_argument('--learning_rate', default=configs['hyperparameters']['learning_rate'], type=float)

    parser.add_argument('--train_path', default=configs['path']['train_path'], type=str)
    parser.add_argument('--val_path', default=configs['path']['val_path'], type=str)
    parser.add_argument('--dev_path', default=configs['path']['dev_path'], type=str)
    parser.add_argument('--predict_path', default=configs['path']['predict_path'], type=str)
    parser.add_argument('--output_path', default=configs['path']['output_path'], type=str)
    parser.add_argument('--checkpoint_path', default=configs['path']['checkpoint_path'], type=str)

    parser.add_argument('--ensemble_list', default=configs['model']['ensemble_weight'], type=list)

    model_list = [i for i in configs['model'].values() if isinstance(i, str)]

    args = parser.parse_args(args=[])

    return args, model_list

if __name__ == '__main__':
    args, model_list = set_parser_and_model()

    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*TensorBoard support*")
    warnings.filterwarnings("ignore", ".*target is close to zero*")

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    for model in model_list:
        print("Train Start With Model Name : ", model)
        args.model_name = model

        train(args)
        inference(args)

    # Ensemble
    print("Starting ensemble process...")
    ensemble(args)  # Ensure this calls the correct function in ensemble.py
