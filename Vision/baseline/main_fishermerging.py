import torch
import os
import json
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from merge_args import parse_arguments
from utils import *
from baseline.fishermerging_utils import *

# Config
args = parse_arguments()
args.save = f'{args.save}/{args.model}'
pretrained_checkpoint = f'{args.save}/zeroshot.pt'
dataset_list = ['MNIST', 'EuroSAT', 'GTSRB',
                'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']


if __name__ == '__main__':
    assert args.val_shot is not None
    dir_path = f'./results/fishermerging/{args.model}/{args.val_shot}/{args.fisher_scaling_coefficient}'

    os.makedirs(dir_path, exist_ok=True)
    res_file_path = f'{dir_path}/acc.json'

    image_encoders_arr = load_image_encoders(args)
    image_encoder = torch.load(pretrained_checkpoint)

    dataloader_arr = []
    for dataset in dataset_list:
        dataset = get_dataset(
            dataset + 'ValfromTrain',
            image_encoder.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            val_fraction=args.val_frac,
            val_shot=args.val_shot,
            seed=args.seed
        )
        dataloader_arr.append(dataset.val_loader)

    exclude_param_names_regex = ["^(?!.*visual).*|(.*conv.*)"]

    merged_param = fisher_merging(image_encoders_arr, dataloader_arr, exclude_param_names_regex,
                    args.fisher_scaling_coefficient, args.device)
    
    copy_params_to_model(merged_param, image_encoder)

    metrics = {}
    for dataset in dataset_list:
        metrics[dataset] = eval_single_dataset(
            image_encoder, dataset + 'ValfromTrain', args, dataset, args.val_frac, args.val_shot)['top1']
    metrics['avg'] = sum(metrics.values()) / len(metrics)
    with open(res_file_path, 'w') as f:
        json.dump(metrics, f)
    print(metrics)

    if args.save_model:
        save_path = f'./saved_models'
        os.makedirs(save_path, exist_ok=True)
        torch.save(image_encoder, f'{save_path}/fishermerging.pt')