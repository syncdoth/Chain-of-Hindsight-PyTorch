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
