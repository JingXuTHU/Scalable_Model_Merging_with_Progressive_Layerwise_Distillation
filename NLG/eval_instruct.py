import json
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import io
import signal
from vllm import SamplingParams
import jsonlines
from utils.evaluate_llms_utils import *
import logging
from typing import Iterable, Dict
import torch
from model_merging_methods.distill_merging_utils import *

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def eval_instruct(llm, args):
    def handler(signum, frame):
        raise TimeoutError("TLE")

    def capture_output_with_timeout(code_string, timeout=2):
        signal.signal(signal.SIGALRM, handler)
        output_capture = io.StringIO()
        
        sys.stdout = output_capture
        
        try:
            signal.alarm(timeout)
            exec(code_string)
            signal.alarm(0)
        except TimeoutError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)
        finally:
            sys.stdout = sys.__stdout__
        
        output = output_capture.getvalue()
        
        return output, None

    os.environ["WANDB_DISABLED"] = "true"

    def load_list_data(data_path):
        # data is a list of dictionaries
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data


    def test_alpaca_eval(llm, finetuned_model_name, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
                        save_gen_results_folder=None):

        eval_set = load_list_data(f"math_code_data/alpaca_eval.json")

        instructions = []
        reference_outputs = []
        for example in eval_set:
            # dictionary with 'instruction', 'output': 'generator' and 'dataset' as keys
            instructions.append(example["instruction"])
            reference_outputs.append(example)

        instructions = instructions[start_index:end_index]
        reference_outputs = reference_outputs[start_index:end_index]

        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)
        logger.info(f"sampling params is {sampling_params}")

        # shutil.rmtree(save_gen_results_folder, ignore_errors=True)
        os.makedirs(save_gen_results_folder, exist_ok=True)
        generator_name = finetuned_model_name
        logger.info(f"generator name is {generator_name}")

        batch_size = 64
        for i in range(0, len(instructions), batch_size):
            batch_instructions = instructions[i:i + batch_size]
            for j in range(len(batch_instructions)):
                batch_instructions[j] = generate_instruction_following_task_prompt(instruction=batch_instructions[j], is_chat_model=True)
            batch_reference_outputs = reference_outputs[i:i + batch_size]
            output_file = f"{save_gen_results_folder}/{start_index + i}.jsonl"

            with torch.no_grad():
                completions = llm.generate(batch_instructions, sampling_params)
            generated_outputs = []
            for idx, output in enumerate(completions):
                generated_text = output.outputs[0].text
                generated_outputs.append({
                    "instruction": batch_reference_outputs[idx]["instruction"],
                    "output": generated_text,
                    "generator": generator_name,
                    "dataset": batch_reference_outputs[idx]["dataset"]
                })
            
            with jsonlines.open(output_file, "w") as writer:
                writer.write_all(generated_outputs)


    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f"test.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    test_alpaca_eval(llm, args.model_path, args, logger, save_gen_results_folder=f"save_gen_results/{args.model_path}")