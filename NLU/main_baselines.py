from utils.load_config import cache_dir
from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed
from utils.customized_trainers import CustomizedTrainer
from utils.metrics import compute_metrics
from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import transformers
import torch
import logging
import time
from functools import partial
import argparse
import sys
import copy
import os
import json

os.environ["WANDB_DISABLED"] = "true"


parser = argparse.ArgumentParser("Interface for merging PLMs on glue")
parser.add_argument("--language_model_name", type=str, default="bert-base-uncased",
                    help="name of the language model", choices=["bert-base-uncased", "roberta-base"])
parser.add_argument("--merging_method_name", type=str, default="distill_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "ties_merging", "adamerging"])
parser.add_argument("--val_shot", type=int, default=64,
                    help="number of examples sampled from training set for validation")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")

# task arithmetic & ties merging args
parser.add_argument("--scaling_coefficient", type=float, default=1.0,
                    help="scaling coefficient to merge the task vector")

# fisher merging args
parser.add_argument("--nums_fisher_examples", type=int, default=64,
                    help="numbers of examples to compute fisher weights")
parser.add_argument("--fisher_scaling_coefficients", type=float, default=1.0,
                    help="scaling coefficients to merge fisher weights")
parser.add_argument("--normalize_fisher_weight", action="store_true",
                    default=False, help="whether to normalize fisher weights (L2 norm) or not")
parser.add_argument("--minimal_fisher_weight", type=float, default=1e-6,
                    help="the minimal value in fisher weights, used for tackling the potential numerical issues")

# regmean merging args
parser.add_argument("--nums_regmean_examples", type=int, default=64,
                    help="numbers of examples to compute regmean weights")
parser.add_argument("--reduce_non_diagonal_ratio", type=float, default=1.0,
                    help="reduce non-diagonal elements in regmean weights by multiplying this scalar")

# ties merging args
parser.add_argument("--param_value_mask_rate", type=float, default=0.8,
                    help="mask rate of the smallest-magnitude parameter values")

# adamerging args
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--granularity", type=str, default="elementwise",
                    help="granularity of the loss", choices=["elementwise", "layerwise", "taskwise"])


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

    save_merge_log_path = f"./save_merge_logs/{args.merging_method_name}/{args.language_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(
        f"{save_merge_log_path}/{args.lr}_{args.epochs}_{args.granularity}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    """
    get the performance of merging method named merging_method_name
    :param args: ArgumentParser, input argument parser
    :param models_to_merge: list, individual models that need to be merged
    :param trainers: list, trainers of individual models
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :param tokenizer: AutoTokenizer, tokenizer
    :return:
    """
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)

    pre_merged_models = copy.deepcopy(models_to_merge)

    # exclude parameter whose name matches "classifier"
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[
                                                       ".*classifier.*"],
                                                   trainers=trainers,
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   nums_fisher_examples=args.nums_fisher_examples,
                                                   fisher_scaling_coefficients=args.fisher_scaling_coefficients,
                                                   normalize_fisher_weight=args.normalize_fisher_weight,
                                                   minimal_fisher_weight=args.minimal_fisher_weight,
                                                   nums_regmean_examples=args.nums_regmean_examples,
                                                   reduce_non_diagonal_ratio=args.reduce_non_diagonal_ratio,
                                                   param_value_mask_rate=args.param_value_mask_rate,
                                                   lr=args.lr,
                                                   epochs=args.epochs,
                                                   granularity=args.granularity
                                                   )

    merged_model_training_args = TrainingArguments(
        output_dir=args.save_merged_model_path,  # save model directory
        # batch size per device during training
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
    )

    acc = {}

    models_to_merge = pre_merged_models

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        # since the classifier is not merged, we additionally set the classifier of merged_model for each model_to_merge
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[
                                    dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(
            f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(
            v, float) else v for k, v in test_metrics.items()}
        logger.info(
            f"test performance on dataset {dataset_name}: {test_metrics}")
        acc[dataset_name] = test_metrics[f'eval_{glue_data_metrics_map[dataset_name]}']

    acc['avg'] = sum(acc.values()) / len(acc)
    logger.info(f"average performance: {acc['avg']}")

    return acc


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

    load_model_paths = []
    for dataset_name in args.dataset_names:
        # best checkpoint setting
        learning_rate = dataset_model_learning_rate_mapping_dict[
            f"{dataset_name}_{args.language_model_name}"]
        load_model_paths.append(
            f"./save_models/{dataset_name}/{args.language_model_name}_lr{learning_rate}")

    args.save_merged_model_path = f"./save_merge_models/{args.merging_method_name}/{args.language_model_name}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name))
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

    # load the checkpoint of each individual model that needs to be merged
    models_to_merge, trainers, = [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):

        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                             val_shot_from_train=args.val_shot,
                                                                                             max_seq_length=128)
        training_args = TrainingArguments(
            output_dir=load_model_path,                        # load model directory
            # batch size per device during training
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        )

        # print(os.path.join(training_args.output_dir, "trainer_state.json"))
        assert os.path.exists(os.path.join(
            training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
        # train_dir = os.path.join(training_args.output_dir, "trainer_state.json")
        # best_dir = json.load(open(train_dir))["best_model_checkpoint"]
        model_to_merge = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=training_args.output_dir,
                                                                            num_labels=num_labels, output_hidden_states=True).to(args.device)
        trainer = CustomizedTrainer(
            model=model_to_merge,               # model to be merged
            args=training_args,                 # training arguments
            # use the validation dataset in the distill merging algorithm
            # training dataset
            train_dataset=val_dataset if args.merging_method_name in [
                "distill_merging", "distill_merging_sequential", "adamerging"] else train_dataset,
            eval_dataset=test_dataset,          # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[
                                    dataset_name]),   # function for computing metrics
            tokenizer=tokenizer                 # tokenizer
        )
        models_to_merge.append(model_to_merge)
        trainers.append(trainer)

    merging_method = MergingMethod(
        merging_method_name=args.merging_method_name, language_model_name=args.language_model_name)

    logger = set_logger(args=args)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")

    best_target_performance = {}
    if args.merging_method_name == "average_merging":
        target_performance = get_merge_performance(
            args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
        logging.info(
            f"target_performance on average_merging: {target_performance}")
        save_path = f'./results/{args.language_model_name}/average_merging/acc.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(target_performance, f)

    elif args.merging_method_name == "task_arithmetic":
        scaling_coefficient_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for scaling_coefficient in scaling_coefficient_range:
            args.scaling_coefficient = scaling_coefficient
            target_performance = get_merge_performance(
                args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
            logging.info(
                f"target_performance on task_arithmetic of scaling_coefficient {scaling_coefficient}: {target_performance}")
            save_path = f'./results/{args.language_model_name}/task_arithmetic/{scaling_coefficient}.json'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(target_performance, f)

    elif args.merging_method_name == "adamerging":
        target_performance = get_merge_performance(
            args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
        logging.info(
            f"target_performance on adamerging of lr={args.lr}, epochs={args.epochs}, granularity={args.granularity}: {target_performance}")
        save_path = f'./results/{args.language_model_name}/adamerging_{args.granularity}/{args.val_shot}/{args.lr}_{args.epochs}.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(target_performance, f)

    elif args.merging_method_name == "fisher_merging":
        nums_fisher_examples = args.nums_fisher_examples
        fisher_scaling_coefficients = args.fisher_scaling_coefficients
        args.fisher_scaling_coefficients = [
            args.fisher_scaling_coefficients] * len(args.dataset_names)
        args.nums_fisher_examples = [
            args.nums_fisher_examples] * len(args.dataset_names)
        target_performance = get_merge_performance(
            args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
        logging.info(
            f"target_performance on fisher_merging of fisher_scaling_coefficient {fisher_scaling_coefficients}, nums_fisher_examples {nums_fisher_examples}: {target_performance}")
        save_path = f'./results/{args.language_model_name}/fisher_merging/{nums_fisher_examples}/{fisher_scaling_coefficients}.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(target_performance, f)

    elif args.merging_method_name == "regmean_merging":
        nums_regmean_examples = args.nums_regmean_examples
        args.nums_regmean_examples = [
            args.nums_regmean_examples] * len(args.dataset_names)
        target_performance = get_merge_performance(
            args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
        logging.info(
            f"target_performance on regmean_merging of nums_regmean_examples {nums_regmean_examples}, reduce_non_diagonal_ratio {args.reduce_non_diagonal_ratio}: {target_performance}")
        save_path = f'./results/{args.language_model_name}/regmean_merging/{nums_regmean_examples}/{args.reduce_non_diagonal_ratio}.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(target_performance, f)

    elif args.merging_method_name == "ties_merging":
        target_performance = get_merge_performance(
            args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
        logging.info(
            f"target_performance on ties_merging of scaling_coefficient {args.scaling_coefficient}, param_value_mask_rate {args.param_value_mask_rate}: {target_performance}")
        save_path = f'./results/{args.language_model_name}/ties_merging/{args.scaling_coefficient}_{args.param_value_mask_rate}.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(target_performance, f)
