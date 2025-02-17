## Introduction
This directory contains the code to run the vision experiments in the paper. 

## Preparation
1. Set Up the Project Directory:
    ```bash
    cd Vision
    export PYTHONPATH="$PYTHONPATH:$PWD"
    ```
2. Install Dependencies:
    ```bash
    conda env create -f requirements.yaml
    ```
3. Prepare the datasets and model checkpoints in `Vision/data` and `Vision/checkpoints` respectively. You can use the ones provided in [AdaMerging](https://github.com/EnnengYang/AdaMerging/tree/main). Note that AdaMerging has renamed the validation and test sets for the DTD dataset. Ensure you switch their names in your file paths accordingly.


## Usage
To run the main script for prodistill, use the following command:
```bash
python main_prodistill.py --model "ViT-B-32" --lr "<learning_rate>" --val-shot "<validation_shot>" --epochs "<number_of_epochs>"
```
Replace `<learning_rate>`, `<validation_shot>`, and `<number_of_epochs>` with the desired values.

The implementation of the baselines can be found in `Vision/baseline`.

## Results
The results of the experiments will be saved in the `results` directory. 
