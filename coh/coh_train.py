from dataclasses import dataclass, field

import wandb
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
                          TrainingArguments)
from transformers.hf_argparser import HfArgumentParser

from coh.data import CoHDataset, CoHDataArgs
from coh.trainer import CoHTrainArgs, CoHTrainer, EvalCallback, compute_metrics


@dataclass
class ExperimentArgs:
    model_name: str = field(default='EleutherAI/gpt-j-6B')
    # paths
    cache_dir: str = field(default='cache')
    # wandb logging
    project_name: str = "CoH"
    run_name: str = 'CoH-GPT-J-6B'


def main():
    parser = HfArgumentParser([ExperimentArgs, CoHDataArgs, CoHTrainArgs])
    args, data_args, coh_train_args = parser.parse_args_into_dataclasses()

    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    wandb.login()
    wandb.init(project=args.project_name, name=args.run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if 't5' in args.model_name or 't0' in args.model_name or 'bart' in args.model_name:
        raise NotImplementedError('encoder-decoder models are not implemented yet.')
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    coh_config = CoHDataset.get_default_config(data_args.__dict__)
    train_dataset = CoHDataset(coh_config, tokenizer)
    eval_dataset = CoHDataset(coh_config, tokenizer)
    test_dataset = CoHDataset(coh_config, tokenizer)

    ################################################################

    training_args = TrainingArguments(**coh_train_args.__dict__)

    trainer = CoHTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EvalCallback(test_dataset, wandb, training_args, tokenizer)],
    )
    trainer.train()


if __name__ == "__main__":
    main()
