from utils.utils import set_random_seed
from utils.customized_trainers import CustomizedTrainer
from utils.llm_data_loader import LLMDataLoader
from transformers import AutoTokenizer, TrainingArguments
import torch
import logging
import argparse
import gc
import sys
import os
import tqdm
import torch.nn.functional as F
import shutil
import time

from model_merging_methods.distill_merging_utils import *

cache_dir = "./MergeLM_models"

os.environ["WANDB_DISABLED"] = "true"

task_model_mapping_dict = {
    "instruct": "WizardLM-13B-V1.2_32001",
    "math": "WizardMath-13B-V1.0_32001",
    "code": "llama-2-13b-code-alpaca_32001"
}
finetuned_model_backbone_mapping_dict = {
    "WizardLM-13B-V1.2_32001": "Llama-2-13b-hf_32001",
    "WizardMath-13B-V1.0_32001": "Llama-2-13b-hf_32001",
    "llama-2-13b-code-alpaca_32001": "Llama-2-13b-hf_32001"
}
finetuned_models = ["WizardLM-13B-V1.2_32001", "WizardMath-13B-V1.0_32001", "llama-2-13b-code-alpaca_32001"]


parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--do_instruct", action="store_true", help="whether to merge instruct model")
parser.add_argument("--do_math", action="store_true", help="whether to merge math model")
parser.add_argument("--do_code", action="store_true", help="whether to merge code model")
parser.add_argument("--language_model_name", type=str,
                    default="Llama-2-13b-hf_32001", help="name of the language model")
parser.add_argument("--merging_method_name", type=str,
                    default="sequential_efficient")
parser.add_argument("--val_shot", type=int, default=32,
                    help="number of examples sampled from training set for validation")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")

parser.add_argument("--tag", type=str,
                    default='test', help="tag for distill merging")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--layer_save", type=str, default="./save_layers", help="path to save layers in merging")
parser.add_argument("--llm_version", type=str, default="v1.0", help="version of the language model")
try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available(
    ) and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()

def check_gpu():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 2:.2f} MB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Cached memory (reserved): {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
        print()


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"train_{args.val_shot}.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def train(args, lr, epochs, merged_train_loader, load_model_paths):
    # Llama-2-13b-hf is the backbone model for all the models

    num_layers = 40

    check_gpu()

    avg_pre_merged_model = load_avg_merged_model_pre_llm(args, merge_coef=0.5)

    check_gpu()

    per_models = []
    for dataset in args.dataset_names:
        fine_tuned_model = load_single_merged_model_pre_llm(args, dataset)
        per_models.append(fine_tuned_model)

    check_gpu()

    merged_train_loader = transform_data_loader_prelayer_pertask_llm(
        merged_train_loader, avg_pre_merged_model, per_models, args.device)
    
    pre_causal_mask = []
    pre_position_ids = []

    pre_causal_mask.append(avg_pre_merged_model.causal_mask)
    pre_position_ids.append(avg_pre_merged_model.position_ids)

    for model in per_models:
        pre_causal_mask.append(model.causal_mask)
        pre_position_ids.append(model.position_ids)

    del avg_pre_merged_model, per_models
    torch.cuda.empty_cache()
    
    check_gpu()

    print('Start training')
    for layer_idx in range(num_layers):
        print(f'Training Layer {layer_idx}')
        merged_layer, layers = load_merged_layers_llm(args=args, layer_idx=layer_idx)
        merged_layer.train()
        for layer in layers:
            layer.eval()
        optimizer = torch.optim.Adam(merged_layer.parameters(), lr=lr)

        for epoch in tqdm.tqdm(range(epochs)):
            total_loss = 0
            for data in merged_train_loader:
                causal_mask = [pre_causal_mask[i].clone() for i in range(len(pre_causal_mask))]
                position_ids = [pre_position_ids[i].clone() for i in range(len(pre_position_ids))]
                x = data['data'].to(args.device)
                # print(f'X: {x.shape}')
                batch_size = x.shape[0]
                x = x.permute(1, 0, 2, 3)

                source_loader = data['source_loader'].to(args.device)

                optimizer.zero_grad()

                feature = merged_layer.get_merged_model()(x[0], causal_mask[0], position_ids[0])[0].reshape(batch_size, -1)

                loss = 0
                idx = source_loader.item()
                with torch.no_grad():
                    true_feature = layers[idx](x[1], causal_mask[idx], position_ids[idx])[0].reshape(batch_size, -1)
                loss += F.mse_loss(feature, true_feature, reduction='none').sum()
                total_loss += loss.detach().clone().cpu().item()
                loss.backward()
                optimizer.step()
            logger.info(f'Layer {layer_idx + 1}/{num_layers}, Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss}')
        print(f'Layer {layer_idx + 1}/{num_layers} finished')
        merged_train_loader = transform_data_loader_layer_pertask_llm(
            merged_train_loader, merged_layer.get_merged_model(), layers,
            args.device, pre_causal_mask, pre_position_ids)
        os.makedirs(
            f'{args.layer_save}/{args.dataset_name_combined}/{args.merging_method_name}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}/{args.llm_version}', exist_ok=True)
        torch.save(
            merged_layer.get_merged_model(), f'{args.layer_save}/{args.dataset_name_combined}/{args.merging_method_name}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}/{args.llm_version}/layer_{layer_idx}.pt')
        del merged_layer, layers
        torch.cuda.empty_cache()

        check_gpu()

    merged_model = load_avg_merged_model_llm(args, merge_coef=0.5)
    for layer_idx in range(num_layers):
        merged_layer = torch.load(
            f'{args.layer_save}/{args.dataset_name_combined}/{args.merging_method_name}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}/{args.llm_version}/layer_{layer_idx}.pt', map_location=args.device)
        for name, _ in merged_model.model.layers[layer_idx].named_parameters():
            set_attr(merged_model.model.layers[layer_idx], name.split('.'), nn.Parameter(get_attr(merged_layer, name.split('.'))))

    shutil.rmtree(f'{args.layer_save}/{args.dataset_name_combined}/{args.merging_method_name}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}/{args.llm_version}')

    return merged_model


if __name__ == "__main__":
    start_time = time.time()

    args.dataset_names = []
    if args.do_instruct:
        args.dataset_names.append("instruct")
    if args.do_math:
        args.dataset_names.append("math")
    if args.do_code:
        args.dataset_names.append("code")
    args.dataset_name_combined = "_".join(args.dataset_names)
    args.cache_dir = cache_dir
    args.task_model_mapping_dict = task_model_mapping_dict
    args.finetuned_model_backbone_mapping_dict = finetuned_model_backbone_mapping_dict
    args.finetuned_models = finetuned_models

    set_random_seed(seed=0)

    load_model_paths = []
    for dataset_name in args.dataset_names:
        # best checkpoint setting
        load_model_paths.append(
            f"./MergeLM_models/{task_model_mapping_dict[dataset_name]}")

    args.save_merged_model_path = f"./save_merge_models/{args.dataset_name_combined}/{args.merging_method_name}/{args.language_model_name}/{args.epochs}/{args.lr}/{args.val_shot}/{args.llm_version}"
    args.load_model_paths_dict = {
        args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(os.path.join(cache_dir, args.language_model_name), args.llm_version))
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm_data_loader = LLMDataLoader(tokenizer=tokenizer)

    check_gpu()

    model_to_merge = load_pretrained_model(args)

    check_gpu()

    trainers, eval_datasets = [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):

        train_dataset, test_dataset = llm_data_loader.load_dataset(dataset_name=dataset_name, max_seq_length=512, val_shot=args.val_shot)

        trainer = CustomizedTrainer(
            model=model_to_merge,
            args=TrainingArguments(
                args.save_merged_model_path,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
            ),
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )

        trainers.append(trainer)
        eval_datasets.append(test_dataset)

    print(llm_data_loader.max_len)

    check_gpu()

    merged_train_loader = merge_data_loaders_from_trainers(trainers)

    for trainer in trainers:
        trainer.model = None
        del trainer

    del trainers

    del model_to_merge

    gc.collect()

    torch.cuda.empty_cache()

    check_gpu()

    logger.info(f"********** Run starts. **********")
    logger.info(f"configuration is {args}")

    check_gpu()

    merged_model = train(args, args.lr, args.epochs,
                         merged_train_loader, load_model_paths)
    
    end_time = time.time()

    logger.info(f"Run finished in {end_time - start_time} seconds with val shot {args.val_shot}")

    os.makedirs(
        args.save_merged_model_path, exist_ok=True)

    merged_model.save_pretrained(args.save_merged_model_path)
    tokenizer.save_pretrained(args.save_merged_model_path)
    
    # save eval_datasets
    for dataset_name, eval_dataset in zip(args.dataset_names, eval_datasets):
        indices = eval_dataset.indices
        torch.save(indices, f"{args.save_merged_model_path}/{dataset_name}_indices.pt")
