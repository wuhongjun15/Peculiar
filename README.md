### This is the online repository of the ISSRE2021 paper titled "Peculiar: Smart Contract Vulnerability DetectionBased on Crucial Data Flow Graph and Pre-trainingTechniques".
## Task Definition

Detect reentrancy vulnerabilities in smart contract.

## Dataset

The dataset we use is [SmartBugs Wild Dataset](https://github.com/smartbugs/smartbugs-wild/tree/master/contracts) and filtered following the paper [Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts](https://arxiv.org/abs/1910.10601).

The tools analysis results we use is [Vulnerability Analysis of Smart Contracts using SmartBugs](https://github.com/smartbugs/smartbugs-results) and filtered following the paper [Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts](https://arxiv.org/abs/1910.10601).

### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one contract.  One row is illustrated below.

   - **contract:** the smart contract

   - **idx:** index of the contract
  
   - **address:** the smart contract's address
  
2. dataset/sol_map_contracts.jsonl is stored in jsonlines format. Each line in the uncompressed file represents the vulnerablities information about one or more contreacts in one solidity file found by different tools.  One row is illustrated below.

   - **address:** the solidity file address
  
   - **tools:** the static analysis tools and their detection information 
     - **<tool_name>:** the name of tool
       - **<contract_name>:** the name of contracts in solidity file
       - **<flag\>:** indicates whether this contract is detected as vulnerable by the tool 
  
 

3. train.txt/valid.txt/test.txt provide examples, stored in the following format:    **idx	label**

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  40,742   |
| Dev   |  20,372   |
| Test  |  142,599  |

You can get data using the following command.

```
unzip dataset.zip
```

## Evaluator

We provide a script to evaluate predictions for this task, and report F1 score

### Example

First you shoulde generate the several tools' analysis results to be compared
```bash
python3 evaluator/tool_analyze.py
```

Then you can use this command to compare these tools with our model
```bash
python3 evaluator/evaluator.py -a dataset/test.txt -p evaluator/tool_analysis_result/honeybadger_test.txt saved_models/predictions.txt

python3 evaluator/evaluator.py -a dataset/test.txt -p evaluator/tool_analysis_result/manticore_test.txt saved_models/predictions.txt

python3 evaluator/evaluator.py -a dataset/test.txt -p evaluator/tool_analysis_result/mythril_test.txt saved_models/predictions.txt

python3 evaluator/evaluator.py -a dataset/test.txt -p evaluator/tool_analysis_result/osiris_test.txt saved_models/predictions.txt

python3 evaluator/evaluator.py -a dataset/test.txt -p evaluator/tool_analysis_result/securify_test.txt saved_models/predictions.txt

python3 evaluator/evaluator.py -a dataset/test.txt -p evaluator/tool_analysis_result/oyente_test.txt saved_models/predictions.txt

python3 evaluator/evaluator.py -a dataset/test.txt -p evaluator/tool_analysis_result/slither_test.txt saved_models/predictions.txt

python3 evaluator/evaluator.py -a dataset/test.txt -p evaluator/tool_analysis_result/smartcheck_test.txt saved_models/predictions.txt
```

evaluator/tool_analysis_result/oyente_test.txt
> {'Recall': 0.5403728855624781, 'Prediction': 0.7618502280591382, 'F1': 0.5684974753934876, 'Accuracy': 0.991870876531574}


saved_models/predictions.txt
> {'Recall': 0.7999323645230136, 'Prediction': 0.9433867621815506, 'F1': 0.8575927421860436, 'Accuracy': 0.9961121583411876}

### Dependency

- python version: python3.6.9
- pip3 install torch
- pip3 install transformers
- pip3 install tree_sitter
- pip3 sklearn


### vulnerability detection

```shell
cd parser
bash build.sh
cd ..
python detect.py dev
```

### Evaluation

```shell
python3 evaluator/evaluator.py -a dataset/test.txt -p saved_models/predictions.txt 2>&1| tee saved_models/score.log
```

## Result

The results on the test set are shown as below:

| Method      |  Recall   | Precision |    F1     |
| ----------- | :-------: | :-------: | :-------: |
| Honeybadger |   0.505   |   0.872   |   0.509   |
| Manticore   |   0.500   |   0.497   |   0.499   |
| Mythril     |   0.517   |   0.502   |   0.497   |
| osiris      |   0.538   |   0.590   |   0.553   |
| Oyente      |   0.541   |   0.656   |   0.564   |
| securify    |   0.548   |   0.526   |   0.534   |
| Slither     |   0.654   |   0.520   |   0.526   |
| smartcheck  |   0.705   |   0.794   |   0.741   |
| DR-GCN      |   0.809   |   0.724   |   0.764   |
| TMP         |   0.826   |   0.741   |   0.781   |
| Peculiar    | **0.924** | **0.918** | **0.921** |
