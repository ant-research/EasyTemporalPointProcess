import math
from typing import Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader

from easy_tpp.preprocess.data_collator import TPPDataCollator
from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.utils import py_assert


class TPPDataset(Dataset):
    def __init__(self, data: Dict):
        self.data_dict = data
        self.time_seqs = self.data_dict['time_seqs']
        self.time_delta_seqs = self.data_dict['time_delta_seqs']
        self.type_seqs = self.data_dict['type_seqs']

    def __len__(self):
        """

        Returns: length of the dataset

        """

        py_assert(len(self.time_seqs) == len(self.type_seqs) and len(self.time_delta_seqs) == len(self.type_seqs),
                  ValueError,
                  f"Inconsistent lengths for data! time_seq_len:{len(self.time_seqs)}, event_len: "
                  f"{len(self.type_seqs)}, time_delta_seq_len: {len(self.time_delta_seqs)}")

        return len(self.time_seqs)

    def __getitem__(self, idx):
        """

        Args:
            idx: iteration index

        Returns:
            dict: a dict of time_seqs, time_delta_seqs and type_seqs element

        """
        return dict({'time_seqs': self.time_seqs[idx], 'time_delta_seqs': self.time_delta_seqs[idx],
                     'type_seqs': self.type_seqs[idx]})

    def get_dt_stats(self):
        all_dts = np.concatenate([
            np.array(dts[1:-1 if marks[-1] == -1 else None])
            for dts, marks in zip(self.time_delta_seqs, self.type_seqs)
        ])
        min_dt, max_dt = all_dts.min(), all_dts.max()
        log_dts = np.log(np.clip(all_dts, 1e-5, None))  # matches inter_times.clamp(min=1e-5) in IntensityFree.loglike_loss
        return log_dts.mean(), log_dts.std(), min_dt, max_dt


def get_data_loader(dataset: TPPDataset, backend: str, tokenizer: EventTokenizer, **kwargs):
    assert backend == 'torch', 'Only torch backend is supported.'
    padding = True if tokenizer.padding_strategy is None else tokenizer.padding_strategy
    truncation = False if tokenizer.truncation_strategy is None else tokenizer.truncation_strategy
    data_collator = TPPDataCollator(tokenizer=tokenizer,
                                    return_tensors='pt',
                                    max_length=tokenizer.model_max_length,
                                    padding=padding,
                                    truncation=truncation)
    return DataLoader(dataset,
                      collate_fn=data_collator,
                      **kwargs)
