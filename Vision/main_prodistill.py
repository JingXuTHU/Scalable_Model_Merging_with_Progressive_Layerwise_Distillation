import torch
import os
import json
import tqdm
import shutil
import torch.nn.functional as F
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from merge_args import parse_arguments
from merge_datasets import merge_data_loaders, transform_data_loader_per_task_pre_resblock, transform_data_loader_per_task
from utils import *

# Config
args = parse_arguments()
args.save = f'{args.save}/{args.model}-official'
args.layer_save = f'{args.layer_save}/{args.model}'
pretrained_checkpoint = f'{args.save}/zeroshot.pt'
dataset_list = ['MNIST', 'EuroSAT', 'GTSRB',
                'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']


def train(lr, epochs, val_frac, val_shot):
    if args.model in ['ViT-B-32', 'ViT-B-16']:
        num_layers = 12
    elif args.model in ['ViT-L-14']:
        num_layers = 24
    else:
        raise ValueError(f'Invalid model: {args.model}')

    avg_merged_model = get_avg_merged_model(args, merge_coef = args.backbone_merging_coef)

    dataloader_arr = []
    for dataset in dataset_list:
        dataset = get_dataset(
            dataset + 'ValfromTrain',
            avg_merged_model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            val_fraction=val_frac,
            val_shot=val_shot,
            seed=args.seed
        )
        dataloader_arr.append(dataset.val_loader)

    merged_dataloader = merge_data_loaders(
        dataloader_arr, batch_size=args.batch_size)
        
    preresblock_activation_extractor = create_preresblock_activation_extractor(
        avg_merged_model)
    del avg_merged_model
    torch.cuda.empty_cache()
    
    per_task_preresblock_activation_extractors = []
    for dataset in dataset_list:
        image_encoder = torch.load(
            f'{args.save}/{dataset}/finetuned.pt', map_location=args.device)
        per_task_preresblock_activation_extractors.append(
            create_preresblock_activation_extractor(image_encoder))
        del image_encoder
        torch.cuda.empty_cache()

    merged_dataloader = transform_data_loader_per_task_pre_resblock(
        merged_dataloader, preresblock_activation_extractor, per_task_preresblock_activation_extractors, args.device)
    del preresblock_activation_extractor, per_task_preresblock_activation_extractors
    torch.cuda.empty_cache()

    print('Start training')
    for layer_idx in range(num_layers):
        print(f'Training Layer {layer_idx}')
        merged_layer, layers = load_merged_layers(args, layer_idx)
        merged_layer.train()
        for layer in layers:
            layer.eval()
        optimizer = torch.optim.Adam(merged_layer.parameters(), lr=lr)

        for _ in tqdm.tqdm(range(epochs)):
            for data in merged_dataloader:
                # shape x: (batchsize, 2, seq_len, num_features)
                x = data['data'].to(args.device)
                
                batch_size = x.shape[0]
                # the input to each layer should be of shape (seq_len, batch_size, hidden_size)
                x = x.permute(1, 2, 0, 3)

                source_loader = data['source_loader'].to(args.device)
                optimizer.zero_grad()

                loss = 0
                # the output shape of each layer is (seq_len, batch_size, hidden_size)
                feature = merged_layer.get_merged_model()(x[0]).permute(1, 0, 2).reshape(batch_size, -1)

                for idx in range(len(dataset_list)):
                    with torch.no_grad():
                        true_feature = layers[idx](x[1]).permute(1, 0, 2).reshape(batch_size, -1)

                    loss_dataset = F.mse_loss(
                        feature, true_feature, reduction='none').sum(dim=1)
                    loss_dataset = (
                        loss_dataset * source_loader.eq(idx).float()).sum()
                    loss += loss_dataset

                loss.backward()
                optimizer.step()

        merged_dataloader = transform_data_loader_per_task(
            merged_dataloader, merged_layer, layers, args.device)
        os.makedirs(
            f'{args.layer_save}/{args.epochs}/{args.lr}/{args.val_shot}', exist_ok=True)
        torch.save(
            merged_layer.get_merged_model(), f'{args.layer_save}{args.epochs}/{args.lr}/{args.val_shot}/layer_{layer_idx}.pt')
        del merged_layer, layers
        torch.cuda.empty_cache()

    print('Training finished')

    merged_image_encoder = get_avg_merged_model(args, merge_coef = args.backbone_merging_coef)

    for layer_idx in range(num_layers):
        merged_layer = torch.load(
            f'{args.layer_save}/{args.epochs}/{args.lr}/{args.val_shot}/layer_{layer_idx}.pt', map_location=args.device)
        merged_image_encoder.model.visual.transformer.resblocks[layer_idx] = merged_layer

    shutil.rmtree(f'{args.layer_save}/{args.epochs}/{args.lr}/{args.val_shot}')

    return merged_image_encoder


if __name__ == '__main__':
    print(
        f'lr: {args.lr}, epochs: {args.epochs}, val_shot: {args.val_shot}')

    image_encoder = train(
        args.lr, args.epochs, args.val_frac, args.val_shot)
    
    if args.save_model:
        save_path = f'./saved_models/prodistill'
        os.makedirs(save_path, exist_ok=True)
        torch.save(image_encoder, f'{save_path}/{args.lr}_{args.epochs}.pt')

    dir_path = f'./results/{args.val_shot}'
    os.makedirs(dir_path, exist_ok=True)
    res_file_path = f'{dir_path}/{args.lr}_{args.epochs}.json'

    metrics = {}
    for dataset in dataset_list:
        metrics[dataset] = eval_single_dataset(
            image_encoder, dataset + 'ValfromTrain', args, dataset, args.val_frac, args.val_shot)['top1']
    metrics['avg'] = sum(metrics.values()) / len(metrics)
    with open(res_file_path, 'w') as f:
        json.dump(metrics, f)
    print(metrics)
