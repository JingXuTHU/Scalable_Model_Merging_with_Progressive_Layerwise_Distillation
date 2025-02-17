from utils.load_config import cache_dir
from utils.utils import set_random_seed
from utils.customized_trainers import CustomizedTrainer
from utils.metrics import compute_metrics
from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import torch
import logging
from functools import partial
import argparse
import sys
import os
import json
import tqdm
import torch.nn.functional as F
import shutil
from model_merging_methods.distill_merging_utils import *


os.environ["WANDB_DISABLED"] = "true"


parser = argparse.ArgumentParser("Interface for merging PLMs on glue")
parser.add_argument("--language_model_name", type=str, default="bert-base-uncased",
                    help="name of the language model", choices=["bert-base-uncased", "roberta-base"])
parser.add_argument("--merging_method_name", type=str, default="prodistill")
parser.add_argument("--val_shot", type=int, default=64,
                    help="number of examples sampled from training set for validation")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
parser.add_argument("--seed", type=int, default=0, help="random seed")

parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--layer_save", type=str, default="./save_layers", help="path to save layers in merging")
try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available(
    ) and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()


def set_logger(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    return logger


def train(args, lr, epochs, merged_train_loader, load_model_paths):
    # for both bert and roberta:
    # 1. the output of each encoder layer is a tuple, of shape [batch_size, seq_len, embedding_dim]
    # 2. the input is also a tuple, which contains the output of previous layer, with shape [batch_size, seq_len, embedding_dim],
    # and attention mask of shape [batch_size, 1, seq_len, seq_len], and the other inputs are None, None, None, None, False

    # bert-base-uncased has 12 layers, roberta-base has 12 layers
    num_layers = 12

    avg_merged_model = load_avg_merged_model(args)

    pre_encoder_activation_extractor = create_pre_encoder_activation_extractor(
        avg_merged_model, language_model_name=args.language_model_name)
    del avg_merged_model
    torch.cuda.empty_cache()

    per_task_pre_encoder_activation_extractors = []
    for dataset in args.dataset_names:
        fine_tuned_model = load_fine_tuned_model(args, dataset_name)
        per_task_pre_encoder_activation_extractors.append(
            create_pre_encoder_activation_extractor(fine_tuned_model, language_model_name=args.language_model_name))
        del fine_tuned_model
        torch.cuda.empty_cache()

    merged_train_loader = transform_data_loader_prelayer_pertask(
        merged_train_loader, pre_encoder_activation_extractor, per_task_pre_encoder_activation_extractors, args.device)
    
    del pre_encoder_activation_extractor, per_task_pre_encoder_activation_extractors
    torch.cuda.empty_cache()

    print('Start training')
    for layer_idx in range(num_layers):
        print(f'Training Layer {layer_idx}')
        merged_layer, layers = load_merged_layers(args, layer_idx)
        merged_layer.train()
        for layer in layers:
            layer.eval()
        optimizer = torch.optim.Adam(merged_layer.parameters(), lr=lr)

        for epoch in tqdm.tqdm(range(epochs)):
            for data in merged_train_loader:
                # shape of x: [batch_size, 2, seq_length, embedding_dim] -> [2, batch_size, seq_length, embedding_dim]
                x = data['data'].to(args.device)
                batch_size = x.shape[0]
                x = x.permute(1, 0, 2, 3)

                source_loader = data['source_loader'].to(args.device)

                # shape of attention mask: [batch_size, 2, 1, seq_length, seq_length] -> [2, batch_size, 1, seq_length, seq_length]
                attention_mask = data['attention_mask'].to(args.device)
                attention_mask = attention_mask.permute(1, 0, 2, 3, 4)

                optimizer.zero_grad()

                feature = merged_layer.get_merged_model()(x[0], attention_mask[0], None, None, None, None, False)[0].reshape(batch_size, -1)

                loss = 0

                # the data of the batch have the same source_loader
                # source_loader: tensor([idx])
                idx = source_loader.item()
                with torch.no_grad():
                    true_feature = layers[idx](x[1], attention_mask[1], None, None, None, None, False)[0].reshape(batch_size, -1)
                loss += F.mse_loss(feature, true_feature, reduction='none').sum()

                loss.backward()
                optimizer.step()
        print(f'Layer {layer_idx + 1}/{num_layers} finished')
        merged_train_loader = transform_data_loader_layer_pertask(
            merged_train_loader, merged_layer.get_merged_model(), layers, args.device)
        os.makedirs(
            f'{args.layer_save}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}', exist_ok=True)
        torch.save(
            merged_layer.get_merged_model(), f'{args.layer_save}/{args.language_model_name}//{args.epochs}/{args.lr}/{args.val_shot}/layer_{layer_idx}.pt')
        del merged_layer, layers
        torch.cuda.empty_cache()

    merged_model = load_avg_merged_model(args)
    for layer_idx in range(num_layers):
        merged_layer = torch.load(
            f'{args.layer_save}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}/layer_{layer_idx}.pt', map_location=args.device)
        if args.language_model_name == 'bert-base-uncased':
            merged_model.bert.encoder.layer[layer_idx] = merged_layer
        elif args.language_model_name == 'roberta-base':
            merged_model.roberta.encoder.layer[layer_idx] = merged_layer

    shutil.rmtree(f'{args.layer_save}/{args.language_model_name}/{args.tag}/{args.epochs}/{args.lr}/{args.val_shot}')

    return merged_model


def evaluate(args, merged_model, classifier_heads, eval_datasets, tokenizer, logger):
    acc = {}

    for idx, (dataset_name, classifier, eval_dataset) in enumerate(zip(args.dataset_names, classifier_heads, eval_datasets)):
        # since the classifier is not merged, we additionally set the classifier of merged_model for each model_to_merge
        merged_model.classifier = classifier

        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=TrainingArguments(
                args.save_merged_model_path,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
            ),
            eval_dataset=eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[
                                    dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )
        merged_model.eval()

        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(
            v, float) else v for k, v in test_metrics.items()}
        logger.info(
            f"test performance on dataset {dataset_name}: {test_metrics}")
        acc[dataset_name] = test_metrics[f'eval_{glue_data_metrics_map[dataset_name]}']

    acc['avg'] = sum(acc.values()) / len(acc)
    logger.info(f"average performance: {acc['avg']}")

    logging.info(
        f"target_performance on {args.merging_method_name} of lr={args.lr}, epochs={args.epochs}: {acc}")
    save_path = f'./results/{args.language_model_name}/{args.merging_method_name}/{args.val_shot}/{args.lr}_{args.epochs}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(acc, f)


if __name__ == "__main__":
    args.dataset_names = ["cola", "sst2", "mrpc",
                          "stsb", "qqp", "mnli", "qnli", "rte"]
    dataset_model_learning_rate_mapping_dict = {
        "cola_bert-base-uncased": 1e-5,
        "sst2_bert-base-uncased": 1e-5,
        "mrpc_bert-base-uncased": 1e-5,
        "stsb_bert-base-uncased": 1e-5,
        "qqp_bert-base-uncased": 1e-5,
        "mnli_bert-base-uncased": 1e-5,
        "qnli_bert-base-uncased": 1e-5,
        "rte_bert-base-uncased": 1e-5,
        "cola_roberta-base": 1e-5,
        "sst2_roberta-base": 1e-5,
        "mrpc_roberta-base": 1e-5,
        "stsb_roberta-base": 1e-5,
        "qqp_roberta-base": 1e-5,
        "mnli_roberta-base": 1e-5,
        "qnli_roberta-base": 1e-5,
        "rte_roberta-base": 1e-5
    }

    set_random_seed(seed=args.seed)

    load_model_paths = []
    for dataset_name in args.dataset_names:
        learning_rate = dataset_model_learning_rate_mapping_dict[
            f"{dataset_name}_{args.language_model_name}"]
        load_model_paths.append(
            f"./save_models/{dataset_name}/{args.language_model_name}_lr{learning_rate}")

    args.save_merged_model_path = f"./save_merge_models/{args.merging_method_name}/{args.language_model_name}"
    args.load_model_paths_dict = {
        args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name))
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

    trainers, eval_datasets, nums_labels = [], [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):

        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                             val_shot_from_train=args.val_shot,
                                                                                             max_seq_length=128,
                                                                                             seed=args.seed)

        nums_labels.append(num_labels)

        model_to_merge = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=load_model_path,
                                                                            num_labels=num_labels).to(args.device)

        trainer = CustomizedTrainer(
            model=model_to_merge,
            args=TrainingArguments(
                args.save_merged_model_path,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
            ),
            train_dataset=val_dataset,
            tokenizer=tokenizer
        )

        trainers.append(trainer)
        eval_datasets.append(test_dataset)

    merged_train_loader = merge_data_loaders_from_trainers(trainers)

    del trainers
    torch.cuda.empty_cache()

    logger = set_logger(args=args)
    logger.info(f"********** Run starts. **********")
    logger.info(f"configuration is {args}")

    merged_model = train(args, args.lr, args.epochs,
                         merged_train_loader, load_model_paths)

    classifier_heads = []
    for dataset_idx in range(len(args.dataset_names)):
        load_model_path = load_model_paths[dataset_idx]
        num_labels = nums_labels[dataset_idx]
        model_to_merge = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=load_model_path,
                                                                            num_labels=num_labels).to(args.device)
        classifier_heads.append(model_to_merge.classifier)
        del model_to_merge

    evaluate(args, merged_model, classifier_heads,
             eval_datasets, tokenizer, logger)
