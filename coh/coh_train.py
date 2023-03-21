from dataclasses import dataclass, field

import wandb
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.hf_argparser import HfArgumentParser
import torch

from coh.data import CoHDataset, CoHDataArgs, CoHDataCollator
from coh.trainer import CoHTrainArgs, CoHTrainer, EvalCallback, compute_metrics


@dataclass
class ExperimentArgs:
    model_name: str = field(default='EleutherAI/gpt-j-6B')
    # paths
    cache_dir: str = field(default='cache')
    # wandb logging
    wandb_project_name: str = "CoH"
    wandb_run_name: str = 'CoH-GPT-J-6B'


def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # allows tf32, only on Ampere GPUs
    parser = HfArgumentParser([ExperimentArgs, CoHDataArgs, CoHTrainArgs])
    args, data_args, coh_train_args = parser.parse_args_into_dataclasses()

    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    wandb.login()
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if 't5' in args.model_name or 't0' in args.model_name or 'bart' in args.model_name:
        raise NotImplementedError('encoder-decoder models are not implemented yet.')
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    coh_config = CoHDataset.get_default_config(data_args.__dict__)
    train_dataset = CoHDataset(coh_config, tokenizer)
    eval_cfg = coh_config.copy_and_resolve_references()
    eval_cfg.update({'split': 'validation'})
    eval_dataset = CoHDataset(eval_cfg, tokenizer)
    test_cfg = coh_config.copy_and_resolve_references()
    test_cfg.update({'split': 'test'})
    test_dataset = CoHDataset(test_cfg, tokenizer)

    ################################################################

    trainer = CoHTrainer(
        model=model,
        tokenizer=tokenizer,
        args=coh_train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CoHDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EvalCallback(test_dataset, wandb, coh_train_args, tokenizer)],
    )
    trainer.train()


if __name__ == "__main__":
    main()
