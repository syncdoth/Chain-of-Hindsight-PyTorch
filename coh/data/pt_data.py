"""
Copied from original repo: https://github.com/lhao499/CoH
Adapted by Sehyun Choi, 2023
"""

import torch
from datasets import load_dataset
from ml_collections import ConfigDict


class PretrainDataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.seq_length = 1024
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.field = 'text'
        config.streaming = True
        config.batch_size = 1

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None

        self._tokenizer = tokenizer

        # use validation set as test set too
        split = 'validation' if split == 'test' else split
        self._dataset = load_dataset(self.config.path,
                                     name,
                                     split=split,
                                     streaming=self.config.streaming)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        while True:
            tokens = []
            for example in self._dataset:
                # tokens.append(self.tokenizer.bos_token_id)
                tokens.extend(self.tokenizer.encode(example[self.config.field]))
                tokens.append(self.tokenizer.eos_token_id)
                while len(tokens) > chunk_size:
                    yield {
                        'tokens':
                            torch.LongTensor(tokens[:chunk_size]
                                            ).reshape(self.config.batch_size, -1)
                    }
                    tokens = tokens[chunk_size:]

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
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)
