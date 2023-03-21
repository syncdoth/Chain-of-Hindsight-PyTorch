"""
This combines hf_data and pt_data.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from ml_collections import ConfigDict
from torch.utils.data import IterableDataset
from transformers import DefaultDataCollator

from coh.data.hf_data import HumanFeedbackDataset
from coh.data.pt_data import PretrainDataset


# TODO: better way to us ConfigDict and dataclass together?
@dataclass
class CoHDataArgs:
    seq_length: int = field(
        default=32,
        metadata={"help": "only use the first 32 tokens of documents (including title)"})
    batch_size: int = field(default=512)
    hf_weights: str = field(
        default="",
        metadata={
            "help": "comma-separated weights for sampling from each dataset. Length should be 3."
        })


class CoHDataset(IterableDataset):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.seq_length = 512
        config.split = 'train'
        config.batch_size = 8
        ############## hf ##################
        config.hf = ConfigDict()
        config.hf.seq_length = config.seq_length
        config.hf.split = config.split
        config.hf.batch_size = config.batch_size
        # specific
        config.hf.weight = ""
        ############## pt ##################
        config.pt = ConfigDict()
        config.pt.seq_length = config.seq_length
        config.pt.split = config.split
        config.pt.batch_size = config.batch_size
        # specific
        config.pt.path = 'c4'
        config.pt.name = 'en'
        config.pt.field = 'text'
        config.pt.streaming = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        # TODO: make it possible for num_workers=n
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        super().__init__()
        self.config = self.get_default_config(config)

        self._tokenizer = tokenizer
        self._hf_datset = HumanFeedbackDataset(self.config.hf, tokenizer)
        self._pt_datset = PretrainDataset(self.config.pt, tokenizer)

    def __iter__(self):
        for hf, pt in zip(self._hf_datset, self._pt_datset):
            yield {
                'hf_tokens': hf['tokens'],
                'hf_masks': hf['masks'],
                'pt_tokens': pt['tokens'],
            }

    def __getstate__(self):
        return self.config, self.tokenizer

    def __setstate__(self, state):
        config, tokenizer = state
        self.__init__(config, tokenizer)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def hf_dataset(self):
        return self._hf_datset

    @property
    def pt_dataset(self):
        return self._pt_datset

    @property
    def vocab_size(self):
        return len(self._tokenizer)

class CoHDataCollator(DefaultDataCollator):
    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if len(features) == 1:
            # in most cases, this should be true
            return features[0]

        collated = {k: [] for k in features[0].keys()}
        for x in features:
            for k, v in x.items():
                collated[k].append(v)

        collated = {k: torch.cat(v, dim=0) for k, v in collated.items()}
        return collated
