## Environment
see in environment.yaml

## Datasets
The processed data is in ./math_code_data

## Checkpoints
Download checkpoints in ./MergeLM_models from https://huggingface.co/meta-llama/Llama-2-13b-hf, https://huggingface.co/WizardLMTeam/WizardLM-13B-V1.2, https://huggingface.co/vanillaOVO/WizardMath-13B-V1.0, https://huggingface.co/layoric/llama-2-13b-code-alpaca

## Preprocess Checkpoints
Run
```shell
python resize_model_tokens.py --model_name Llama-2-13b-hf
python resize_model_tokens.py --model_name WizardLM-13B-V1.2
python resize_model_tokens.py --model_name WizardMath-13B-V1.0
python resize_model_tokens.py --model_name llama-2-13b-code-alpaca
python split_model.py --model_name Llama-2-13b-hf_32001
python split_model.py --model_name WizardLM-13B-V1.2_32001
python split_model.py --model_name WizardMath-13B-V1.0_32001
python split_model.py --model_name llama-2-13b-code-alpaca_32001
```

## Modify package
Add 
```python
        self.causal_mask = attention_mask
        self.position_ids = position_ids
        self.past_key_values = past_key_values
        self.output_attentions = output_attentions
        self.use_cache = use_cache

        # judge whether the attribute self has layers
        if not hasattr(self, "layers"):
            return hidden_states
```
in lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py: LlamaModel.forawrd before
```python
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
```

## Training
Merging code and math with val_shot 16, lr 0.1, epochs 100:
```shell
python merge_sequential_llm.py --val_shot 16 --batch_size 16 --lr 0.1 --epochs 100 --do_math --do_code 
```
Merging instruct and math with val_shot 16, lr 0.1, epochs 100:
```shell
python merge_sequential_llm.py --val_shot 16 --batch_size 16 --lr 0.1 --epochs 100 --do_math --do_instruct 
```
Merging instrucr and code with val_shot 16, lr 0.1, epochs 100:
```shell
python merge_sequential_llm.py --val_shot 16 --batch_size 16 --lr 0.1 --epochs 100 --do_instruct --do_code 
```

## Evaluation
Evaluate the model with model path in instruct, math and code benchmarks with the single gpu:
```shell
python eval.py --model_path model_path --gpu 1 --do_instruct --do_math --do_code
```
**Key**:
Instruct benchmark is alpaca_eval. When we use the above shell, then we should run
```shell
python merge_gen.py --path eval_generate_path
```
and use alpaca_eval with gpt-4-turbo api to evaluate (Please first see [alpaca_eval repository](https://github.com/tatsu-lab/alpaca_eval) to install the environment):
```shell
alpaca_eval --model_outputs merge_eval_generate_path --annotators_config weighted_alpaca_eval_gpt4_turbo
```