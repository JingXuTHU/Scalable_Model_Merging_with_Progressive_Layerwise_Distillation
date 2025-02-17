import torch
import os
import json
from utils import get_avg_merged_model
from merge_args import parse_arguments
from src.eval import eval_single_dataset

# Config
args = parse_arguments()
args.save = f'{args.save}/{args.model}'
pretrained_checkpoint = f'{args.save}/zeroshot.pt'
dataset_list = ['MNIST', 'EuroSAT', 'GTSRB',
                'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']


if __name__ == '__main__':
    image_encoder = get_avg_merged_model(args, merge_coef=0.3)

    if args.save_model:
        save_path = f'./saved_models/task_arithmetic'
        os.makedirs(save_path, exist_ok=True)
        torch.save(image_encoder, f'{save_path}/model.pt')

    metrics = {}
    for dataset in dataset_list:
        metrics[dataset] = eval_single_dataset(
            image_encoder, dataset + 'ValfromTrain', args, dataset, args.val_frac, args.val_shot)['top1']
    metrics['avg'] = sum(metrics.values()) / len(metrics)

    dir_path = f'./results/simple_avg/{args.model}/0.3'
    os.makedirs(dir_path, exist_ok=True)
    res_file_path = f'{dir_path}/acc.json'
    with open(res_file_path, 'w') as f:
        json.dump(metrics, f)
    print(metrics)
