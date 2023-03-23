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
    tokenizer_name: str = field(default=None,
                                metadata={"help": "Will default to --model_name."})
    # paths
    cache_dir: str = field(default='cache')
    # wandb logging
    wandb_project_name: str = "CoH"
    wandb_run_name: str = 'CoH-GPT-J-6B'
    # webgpt dataset test size
    webgpt_dataset_test_size: float = field(
        default=0.1,
        metadata={"help": "webgpt_comparisons only have train: need to split."},
        )
    # peft
    use_lora: bool = field(default=True, metadata={"help": "use lora with huggingface peft. You must install loralib and peft."})
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1


def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # allows tf32, only on Ampere GPUs
    parser = HfArgumentParser([ExperimentArgs, CoHDataArgs, CoHTrainArgs])
    args, data_args, coh_train_args = parser.parse_args_into_dataclasses()

    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    if coh_train_args.local_rank == 0:
        wandb.init(project=args.wandb_project_name,
                   name=args.wandb_run_name,
                   config={
                       "experiment": args.__dict__,
                       "data": data_args.__dict__,
                       "train": coh_train_args.__dict__,
                       },
                   )

    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir)
    if 't5' in args.model_name or 't0' in args.model_name or 'bart' in args.model_name:
        raise NotImplementedError('encoder-decoder models are not implemented yet.')
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    if args.use_lora:
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        coh_train_args.gradient_checkpointing = False  # incompatible with lora

    webgpt_data = CoHDataset.load_webgpt_dataset(test_size=args.webgpt_dataset_test_size)
    data_args_dict = data_args.__dict__
    coh_config = CoHDataset.get_default_config(data_args_dict)
    train_dataset = CoHDataset(coh_config, tokenizer, webgpt_data)
    # data_args_dict['split'] = 'validation'
    # eval_cfg = CoHDataset.get_default_config(data_args_dict)
    # eval_dataset = CoHDataset(eval_cfg, tokenizer)
    data_args_dict['split'] = 'test'
    test_cfg = CoHDataset.get_default_config(data_args_dict)
    test_dataset = CoHDataset(test_cfg, tokenizer, webgpt_data)

    ################################################################

    trainer = CoHTrainer(
        model=model,
        tokenizer=tokenizer,
        args=coh_train_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=CoHDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EvalCallback(test_dataset, wandb, coh_train_args, tokenizer)],
    )
    trainer.train()
    model.save_pretrained("llama-7b-coh-lora")
    wandb.finish()


if __name__ == "__main__":
    main()
