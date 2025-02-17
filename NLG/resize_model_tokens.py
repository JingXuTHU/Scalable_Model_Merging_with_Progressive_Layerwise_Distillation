# we want to resize the model tokenizer to 32001 tokens and model's embedding

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from utils.utils import smart_tokenizer_and_embedding_resize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--llm_version", type=str, default="v1.0")
    args = parser.parse_args()

    path = f'./MergeLM_models/{args.model_name}'
    path2 = f'{path}_32001/{args.llm_version}'

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)

    print(tokenizer.vocab_size)

    if not os.path.exists(path2):
        os.makedirs(path2)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=model,
        tokenizer=tokenizer,
    )
    model.generation_config.do_sample = True
    model.save_pretrained(path2)
    tokenizer.save_pretrained(path2)