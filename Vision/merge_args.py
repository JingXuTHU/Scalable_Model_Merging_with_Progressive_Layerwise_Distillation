import os
import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default='./data',
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='ViT-B-32',
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save",
        type=str,
        default='./checkpoints',
        help="checkpoint directory",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--val-shot",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default='elementwise',
    )
    parser.add_argument(
        "--tag", 
        type=str,
        default='test',
    )
    parser.add_argument(
        "--layer-save",
        type=str,
        default='./layer_save',
    )
    parser.add_argument(
        "--backbone-merging-coef", 
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--reduce-non-diagonal-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--fisher-scaling-coefficient",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--save-model", 
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )


    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args
