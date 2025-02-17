import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="save_gen_results/save_merge_models/instruct_math/sequential_efficient/Llama-2-13b-hf_32001/100/0.1/64/v1.0")
    args = parser.parse_args()
    num = 13
    data_list = []
    for i in range(num):
        ind = i * 64
        with open(f"{args.path}/{ind}.jsonl", "r") as f:
            d = f.readlines()
            for line in d:
                data_list.append(json.loads(line))
    print(len(data_list))
    with open(f"{args.path}.json", "w") as f:
        json.dump(data_list, f)