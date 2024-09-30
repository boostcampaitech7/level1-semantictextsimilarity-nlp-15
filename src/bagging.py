import argparse
import os
import yaml

from train import train
from inference import inference
from ensemble import bagging_ensemble

def bagging(args, num_splits):
    # Get Model List from config.yaml
    with open(os.path.join('config.yaml')) as f:
        configs = yaml.safe_load(f)

    model_list = [configs['model'][key] for key in configs['model'] if 'model_name' in key]
    model_name = model_list[-1].split('/')[-1]
    name = args.train_path.split('/')[-1].split('.')[0]

    # Train and Inference
    for i in range(num_splits):
        args.train_path = f'./data/{name}.csv_split_part_{i+1}.csv'
        train(args)
        inference(args, i)
        checkpoint_files = [f for f in os.listdir(args.checkpoint_path) if f.endswith('.ckpt') and model_name in f]

        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files found.")

        latest_checkpoint = max(checkpoint_files, key=lambda x: float(x.split('_')[-1].split('=')[-1].replace('.ckpt', '')))
        os.remove(os.path.join(args.checkpoint_path, latest_checkpoint))
    
    bagging_ensemble(args)
    
    
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

    bagging(args)