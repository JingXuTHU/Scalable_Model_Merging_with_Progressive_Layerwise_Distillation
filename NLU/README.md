## Introduction
This directory contains code for Natural Language Understanding (NLU) tasks in the paper. 


## Preparation
1. Set Up the Project Directory:
    ```bash
    cd NLU
    export PYTHONPATH="$PYTHONPATH:$PWD"
    ```
2. Install Dependencies:
    ```bash
    conda env create -f requirements.yaml
    ```
3. Prepare the model checkpoints. You can follow the steps in [DARE](https://github.com/yule-BUAA/MergeLM) to train the model.

To run the model merging script, use the following command:
```bash
python main_prodistill.py --language_model_name roberta-base --seed <seed> --lr <learning_rate> --val_shot <validation_shot> --epochs <num_epochs>
```

Replace `<seed>`, `<learning_rate>`, `<validation_shot>`, `<num_epochs>` with appropriate values.

You can run the baseline using the `main_baselines.py` file.
