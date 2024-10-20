import math
from typing import Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader

from easy_tpp.preprocess.data_collator import TPPDataCollator
from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.utils import py_assert, is_tf_available


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

    def to_tf_dataset(self, data_collator: TPPDataCollator, **kwargs):
        """Generate a dataset to use in Tensorflow

        Args:
            data_collator (TPPDataCollator): collator to tokenize the event data.

        Raises:
            ImportError: Tensorflow is not installed.

        Returns:
            tf.keras.utils.Sequence: tf Dataset object for TPP data.
        """
        if is_tf_available():
            import tensorflow as tf

            if tf.__version__ >= '2.0':
                tf = tf.compat.v1
                tf.disable_v2_behavior()
        else:
            raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

        class TfTPPDataset(tf.keras.utils.Sequence):
            def __init__(self, time_seqs, time_delta_seqs, type_seqs, **kwargs):
                """Initialize the class.

                Args:
                    batch_size (int): size of batch.
                    shuffle (bool): whether to shuffle the data in each batch.

                """
                self.time_seqs = time_seqs
                self.time_delta_seqs = time_delta_seqs
                self.type_seqs = type_seqs
                self.data_len = len(self.time_delta_seqs)
                self.batch_size = kwargs.pop('batch_size')
                self.shuffle = kwargs.pop('shuffle', False)
                self.idx = np.arange(self.data_len)
                self.kwargs = kwargs

            def __getitem__(self, index):
                # get batch indexes from shuffled indexes
                batch_idx = self.idx[index * self.batch_size:(index + 1) * self.batch_size]
                batch = dict({'time_seqs': [self.time_seqs[i] for i in batch_idx],
                              'time_delta_seqs': [self.time_delta_seqs[i] for i in batch_idx],
                              'type_seqs': [self.type_seqs[i] for i in batch_idx]})

                batch = data_collator(batch, **self.kwargs)
                return batch

            def __len__(self):
                # Denotes the number of batches per epoch
                return math.ceil(self.data_len / self.batch_size)

            def on_epoch_end(self):
                # Updates indexes after each epoch
                self.idx = np.arange(self.data_len)
                if self.shuffle:
                    np.random.shuffle(self.idx)

        return TfTPPDataset(self.time_seqs, self.time_delta_seqs, self.type_seqs, **kwargs)

    def get_dt_stats(self):
        x_bar, s_2_x, n = 0., 0., 0
        min_dt, max_dt = np.inf, -np.inf

        for dts, marks in zip(self.time_delta_seqs, self.type_seqs):
            dts = np.array(dts[1:-1 if marks[-1] == -1 else None])
            min_dt = min(min_dt, dts.min())
            max_dt = max(max_dt, dts.max())
            y_bar = dts.mean()
            s_2_y = dts.var()
            m = dts.shape[0]
            n += m
            # Formula taken from https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
            s_2_x = (((n - 1) * s_2_x + (m - 1) * s_2_y) / (n + m - 1)) + (
                        (n * m * ((x_bar - y_bar) ** 2)) / ((n + m) * (n + m - 1)))
            x_bar = (n * x_bar + m * y_bar) / (n + m)

        print(x_bar, (s_2_x ** 0.5))
        print(f'min_dt: {min_dt}')
        print(f'max_dt: {max_dt}')
        return x_bar, (s_2_x ** 0.5), min_dt, max_dt


def get_data_loader(dataset: TPPDataset, backend: str, tokenizer: EventTokenizer, **kwargs):
    use_torch = backend == 'torch'

    padding = True if tokenizer.padding_strategy is None else tokenizer.padding_strategy
    truncation = False if tokenizer.truncation_strategy is None else tokenizer.truncation_strategy

    if use_torch:
        data_collator = TPPDataCollator(tokenizer=tokenizer,
                                        return_tensors='pt',
                                        max_length=tokenizer.model_max_length,
                                        padding=padding,
                                        truncation=truncation)

        return DataLoader(dataset,
                          collate_fn=data_collator,
                          **kwargs)
    else:
        # we pass to placeholders
        data_collator = TPPDataCollator(tokenizer=tokenizer,
                                        return_tensors='np',
                                        max_length=tokenizer.model_max_length,
                                        padding=padding,
                                        truncation=truncation)

        return dataset.to_tf_dataset(data_collator, **kwargs)
