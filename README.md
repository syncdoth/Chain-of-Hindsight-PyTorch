# Chain of Hindsight in PyTorch & Huggingface Trainer

This is an unofficial implementation of [Chain of Hindsight](https://arxiv.org/abs/2302.02676)
using PyTorch and Huggingface Trainer. The data loading script is directly taken from the original
[repo](https://github.com/lhao499/CoH), and only the training part is re-written using PyTorch.

**The code is not yet tested, so run with caution!!**

## Installation

- For pip,

```bash
pip install -r requirements.txt
```

- For conda,

```bash
conda create -f env.yml
```

## Train

A shell script for training can be found in `train.sh`. To customize command line
arguments, take a look at the arguments dataclasses used in the following
files:

- `coh.coh_train.ExperimentArgs`
- `coh.data.coh_data.CoHDataArgs`
- `coh.trainer.CoHTrainArgs`  (this inherits from `transformers.TrainingArguments`)
