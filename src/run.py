import os
import argparse
import glob

import torch
import yaml
import random
import warnings
import transformers
import pandas as pd

from ensemble import ensemble
from train import train
from inference import inference

def set_parser_and_model():
    with open(os.path.join('config.yaml')) as f:
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

    parser.add_argument('--train_path', default=os.path.join(configs['path']['train_path']), type=str)
    parser.add_argument('--val_path', default=os.path.join(configs['path']['val_path']), type=str)
    parser.add_argument('--dev_path', default=os.path.join(configs['path']['dev_path']), type=str)
    parser.add_argument('--predict_path', default=os.path.join(configs['path']['predict_path']), type=str)
    parser.add_argument('--output_path', default=os.path.join(configs['path']['output_path']), type=str)
    parser.add_argument('--ensemble_output_path', default=os.path.join(configs['path']['ensemble_output_path']), type=str)
    parser.add_argument('--checkpoint_path', default=os.path.join(configs['path']['checkpoint_path']), type=str)

    parser.add_argument('--ensemble_list', default=configs['model']['ensemble_weight'], type=list)
    parser.add_argument('--aug_list', default=configs['aug_list'])

    model_list = [i for i in configs['model'].values() if isinstance(i, str)]
    args = parser.parse_args(args=[])

    return args, model_list

def check_model_existence(args):
    #epoch = args.max_epoch
    model_name = args.model_name.replace("/", "-")

    # find model file by wildcard pattern
    file_pattern = f'model/{model_name}_*.ckpt'
    file_pattern2 = f'checkpoint/{model_name}_*.ckpt'

    matching_files = glob.glob(file_pattern)
    matching_files2 = glob.glob(file_pattern2)

    return matching_files, matching_files2

def print_dataset(args):
    train_dataset = pd.read_csv(args.train_path)
    val_dataset = pd.read_csv(args.val_path)
    dev_dataset = pd.read_csv(args.dev_path)
    predict_dataset = pd.read_csv(args.predict_path)

    # Print dataset's features
    print("Train Dataset")
    print(train_dataset.info())
    print("\nVal Dataset")
    print(val_dataset.info())
    print("\nDev Dataset")
    print(dev_dataset.info())
    print("\nPredict Dataset")
    print(predict_dataset.info())


if __name__ == '__main__':
    args, model_list = set_parser_and_model()
    flag = None

    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*TensorBoard support*")
    warnings.filterwarnings("ignore", ".*target is close to zero*")

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    #print_dataset(args)

    for model in model_list:
        print("Train Start With Model Name : ", model)
        args.model_name = model

        existence, existence_checkpoint = check_model_existence(args)

        if not existence:
            train(args)
            inference(args)

        else:
            # Ask user if they want to retrain the model
            response = None

            if flag is None:
                print(f"Model {model} already exists. Do you want to retrain it? (y/n/all/never)")
                response = input()

                while response.lower() not in ['y', 'n', 'all', 'never']:
                    print("Invalid response. Please enter (y/n/all/never).")
                    response = input()

                if response.lower() == 'never':
                    flag = False

                elif response.lower() == 'all':
                    flag = True

                if response.lower() == 'y':
                    # remove existing model
                    for file in existence:
                        os.remove(file)

                    for file in existence_checkpoint:
                        os.remove(file)

                    train(args)
                    inference(args)

            if flag:
                train(args)
                inference(args)

            else:
                print(f"Skipping model {model}.")
                continue

    # Ensemble
    print("Starting ensemble process...")
    ensemble(args)
