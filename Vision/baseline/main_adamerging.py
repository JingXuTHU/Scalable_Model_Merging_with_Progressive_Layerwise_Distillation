import torch
import os
import json
import tqdm
from src.eval import eval_single_dataset
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.datasets.registry import get_dataset
from merge_args import parse_arguments
from merge_models import MergedModel
from merge_datasets import merge_data_loaders
from utils import *

# Config
args = parse_arguments()
args.save = f'{args.save}/{args.model}'
pretrained_checkpoint = f'{args.save}/zeroshot.pt'
dataset_list = ['MNIST', 'EuroSAT', 'GTSRB',
                'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def train(merged_image_encoder, image_encoders_arr, lr, epochs, val_frac, val_shot):
    merged_image_encoder.train()
    for image_encoder in image_encoders_arr:
        image_encoder.eval()

    dataloader_arr = []
    for dataset in dataset_list:
        dataset = get_dataset(
            dataset + 'ValfromTrain',
            image_encoders_arr[0].val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            val_fraction=val_frac,
            val_shot=val_shot,
            seed=args.seed
        )
        dataloader_arr.append(dataset.val_loader)

    merged_dataloader = merge_data_loaders(
        dataloader_arr, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(merged_image_encoder.parameters(), lr=lr)

    heads = []
    for dataset in dataset_list:
        head = get_classification_head(args, dataset).to(args.device)
        heads.append(head)

    print('Start training')
    for _ in tqdm.tqdm(range(epochs)):
        for data in merged_dataloader:
            data = maybe_dictionarize(data)
            x = data['data'].to(args.device)
            source_loader = data['source_loader'].to(args.device)
            optimizer.zero_grad()

            loss = 0
            feature = merged_image_encoder(x)
            for idx in range(len(image_encoders_arr)):
                logits = heads[idx](feature)
                loss += (softmax_entropy(logits) *
                         source_loader.eq(idx).float()).sum()

            loss.backward()
            optimizer.step()

    print('Training finished')

    return merged_image_encoder.get_merged_model(), merged_image_encoder.get_avg_coef()


if __name__ == '__main__':
    print(f'lr: {args.lr}, epochs: {args.epochs}, val_shot: {args.val_shot}')
    dir_path = f'./results/adamerging/{args.granularity}/{args.val_shot}'

    os.makedirs(dir_path, exist_ok=True)
    res_file_path = f'{dir_path}/{args.lr}_{args.epochs}.json'

    image_encoders_arr = load_image_encoders(args)
    image_encoder_pretrained = torch.load(pretrained_checkpoint)
    merged_image_encoder = MergedModel(
        image_encoder_pretrained, image_encoders_arr, args.granularity).to(args.device)

    image_encoder, coefs = train(
        merged_image_encoder, image_encoders_arr, args.lr, args.epochs, args.val_frac, args.val_shot)

    metrics = {}
    for dataset in dataset_list:
        metrics[dataset] = eval_single_dataset(
            image_encoder, dataset + 'ValfromTrain', args, dataset, args.val_frac, args.val_shot)['top1']
    metrics['avg'] = sum(metrics.values()) / len(metrics)
    metrics['coefs'] = coefs
    with open(res_file_path, 'w') as f:
        json.dump(metrics, f)
    print(metrics)

    if args.save_model:
        save_path = f'./saved_models'
        os.makedirs(save_path, exist_ok=True)
        torch.save(image_encoder, f'{save_path}/adamerging.pt')
