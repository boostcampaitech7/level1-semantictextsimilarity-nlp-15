import pandas as pd
import yaml
import argparse
import torch

def ensemble(args, model_list, sum=False):
    # Get Model List from config.yaml
    flag = None
    epoch = args.max_epoch

    csvs = []

    for model_name in model_list:
        name = model_name.replace("/", "-")
        file_path = f'{args.output_path}/{name}_{epoch}.csv'

        label = pd.read_csv(file_path).iloc[:, -1]
        csvs.append(label)

    # Ensemble
    weights = args.ensemble_list
    torch.set_printoptions(precision=1)

    predictions = torch.stack([torch.Tensor(csv.values) for csv in csvs], dim=1)

    # Do Weighted Mean if sum is False
    if not sum and len(weights) == len(model_list):
        flag = True

        ensemble_predictions = predictions * torch.Tensor(weights)
        ensemble_predictions = ensemble_predictions.sum(dim=1)
        ensemble_predictions = torch.clamp(ensemble_predictions, min=0, max=5)

    # Do Weighted Sum if sum is True
    else:
        flag = False

        ensemble_predictions = torch.sum(predictions, dim=1)
        ensemble_predictions /= len(model_list)
        ensemble_predictions = torch.clamp(ensemble_predictions, min=0, max=5)

    result = list(round(float(elem), 1) for elem in ensemble_predictions)

    # Save Ensemble Result
    name = ""
    for i, elem in enumerate(model_list):
        name += elem.replace("/", "-")
        name += "_" + str(weights[i] if flag else 1) + "_"

    submission = pd.read_csv(args.output_path + "/sample_submission.csv")
    submission['target'] = result

    submission.to_csv(args.output_path + f"/{name}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)

    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--num_labels', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)

    parser.add_argument('--train_path', default='../util/train.csv')
    parser.add_argument('--val_path', default='../util/dev.csv')
    parser.add_argument('--dev_path', default='../util/dev.csv')
    parser.add_argument('--predict_path', default='../util/test.csv')
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--checkpoints', default='checkpoint/')

    args = parser.parse_args(args=[])

    ensemble(args)