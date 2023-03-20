from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer import Trainer


@dataclass
class CoHTrainArgs(TrainingArguments):
    # training args
    learning_rate: float = 5e-4
    warmup_steps: int = 10000
    weight_decay: float = field(default=0, metadata={"help": "typically, set this to 0.01"})
    # NOTE: batch size is 1 because it is already batched in the CoHDataset
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    evaluation_strategy: str = 'steps'
    eval_steps: int = 10000
    max_steps: int = 1000000
    dataloader_drop_last: bool = False
    report_to: str = 'wandb'
    logging_steps: int = 100
    save_strategy: str = 'no'
    fp16: bool = field(
        default=False,
        metadata={
            "help":
                "gives 0/nan loss at some point during training, seems this is a transformers bug."
        })
    dataloader_num_workers: int = 0  # TODO
    gradient_accumulation_steps: int = 1

    # loss weight
    pt_loss_weight: float = field(default=1.0)


class CoHTrainer(Trainer):
    """
    TODO: implement forgetful causal masking (fcm)
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        # hf data
        hf_input_ids = model._shift_right(inputs['hf_tokens'])
        hf_logits = model(input_ids=hf_input_ids).logits
        # [B, T]
        hf_loss = F.cross_entropy(hf_logits, inputs['hf_tokens'], reduction='none').sum(-1)
        hf_loss = (hf_loss * inputs['hf_masks']).mean()
        # pt data
        if self.args.pt_loss_weight > 0:
            pt_input_ids = model._shift_right(inputs['pt_tokens'])
            pt_loss = model(input_ids=pt_input_ids, labels=inputs['pt_tokens']).loss
        else:
            pt_loss = 0

        loss = hf_loss + self.args.pt_loss_weight * pt_loss

        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            hf_input_ids = model._shift_right(inputs['hf_tokens'])
            hf_logits = model(input_ids=hf_input_ids).logits
            hf_loss = F.cross_entropy(hf_logits, inputs['hf_tokens'], reduction='none').sum(-1)
            hf_loss = (hf_loss * inputs['hf_masks']).mean()
        if prediction_loss_only:
            return hf_loss
        # loss, logit, label
        return (hf_loss, hf_logits, inputs['hf_tokens'])


def compute_metrics(eval_preds):
    """Compute token accuracy of greedy decoding"""
    pred = np.argmax(eval_preds.predictions, axis=-1)
    num_correct = (pred == eval_preds.label_ids).sum()
    num_predict = pred.size
    # TODO: maybe also use inputs['hf_masks'] here too?

    return {'accuracy': num_correct / num_predict}


class EvalCallback(TrainerCallback):

    def __init__(self, test_dataset, logger, args, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=1,  # it is already batched in CoHDataset!
            shuffle=False,
            drop_last=False,
            num_workers=0,  # TODO
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        running_hf_loss = 0
        running_pt_loss = 0
        hf_data_count = 0
        pt_data_count = 0

        model = kwargs['model'].eval()
        for inputs in tqdm(self.dataloader, desc='Evaluating on Test Set'):
            with torch.no_grad():
                hf_input_ids = model._shift_right(inputs['hf_tokens'])
                hf_logits = model(input_ids=hf_input_ids).logits
                # [B, T]
                hf_loss = F.cross_entropy(hf_logits, inputs['hf_tokens'], reduction='none').sum(-1)
                hf_loss = (hf_loss * inputs['hf_masks']).mean()
                running_hf_loss += hf_loss.item() * hf_input_ids.shape[0]
                hf_data_count += hf_input_ids.shape[0]
                # pt data
                if self.args.pt_loss_weight > 0:
                    pt_input_ids = model._shift_right(inputs['pt_tokens'])
                    pt_loss = model(input_ids=pt_input_ids, labels=inputs['pt_tokens']).loss
                    running_pt_loss += pt_loss.item() * pt_input_ids.shape[0]
                    pt_data_count += pt_input_ids.shape[0]
                else:
                    pt_loss = 0

        self.logger.log({
            "HF Loss": running_hf_loss / hf_data_count if hf_data_count > 0 else 0,
            "PT Loss": running_pt_loss / pt_data_count if pt_data_count > 0 else 0,
        })
