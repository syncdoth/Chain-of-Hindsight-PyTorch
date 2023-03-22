# Chain of Hindsight in PyTorch & Huggingface Trainer

This is an unofficial implementation of [Chain of Hindsight](https://arxiv.org/abs/2302.02676)
using PyTorch and Huggingface Trainer. The data loading script is directly taken from the original
[repo](https://github.com/lhao499/CoH), and only the training part is re-written using PyTorch.

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

A shell script for training can be found in `train.sh`. It takes gpu device ids
as inputs and passes it to `CUDA_VISIBLE_DEVICES` environment variable.

```bash
sh train.sh 0,1,2,3
```

To customize command line arguments, take a look at the arguments dataclasses
used in the following files:

- `coh.coh_train.ExperimentArgs`
- `coh.data.coh_data.CoHDataArgs`
- `coh.trainer.CoHTrainArgs`  (this inherits from `transformers.TrainingArguments`)

### Train LLaMA

Train script for LLaMA is also provided. The baseline script is:

```bash
sh llama_train.sh 0,1,2,3 ${LLAMA_PATH}
```

To use this script, you will need to have already downloaded LLaMa weights and
converted it to pytorch weights using the convert script at huggingface transformers repo.

- Relevant [PR](https://github.com/huggingface/transformers/pull/21955)

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights \
    --model_size 7B \
    --output_dir /output/path
```
## DeepSpeed

To use DeepSpeed, you need nvcc with the correct version installed. Conda provides
`cuda-nvcc` package, which is also included in `env.yml`. However, to use this,
you need to set the `CUDA_HOME` environment variable to point to the conda environment
(this is required for deepspeed JIT c++ compiler to point to the conda installed
`nvcc` not the system-wide one). after creating the environment and activating it, set

```bash
export CUDA_HOME=/path/to/conda/envs/coh
```

Example deepspeed config files can be found in `ds_config`. They are directly
taken from huggingface's deepspeed integration tutorial.

By default, `llama_train.sh` uses deepspeed, while `train.sh` does not. You can
customize them to suit your needs.

## Notice

This repo diverges from the original repo's implementation in a few ways:
1. The original repo does not have evaluation step.
2. Here, no `bos_token` is prepended to the `input_ids`. This is because since the
   batching logic is chunk-wise, each sentence in a batch is not really a sentence.
3. No `weight_decay_mask` is used.
4. Forgetful Causal Masking is not applied.
