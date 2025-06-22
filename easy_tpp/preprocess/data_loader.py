import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from easy_tpp.preprocess.dataset import TPPDataset
from easy_tpp.preprocess.dataset import get_data_loader
from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.utils import load_pickle, py_assert


class TPPDataLoader:
    def __init__(self, data_config, **kwargs):
        """Initialize the dataloader

        Args:
            data_config (EasyTPP.DataConfig): data config.
            backend (str): backend engine, defaults to 'torch'.
        """
        self.data_config = data_config
        self.num_event_types = data_config.data_specs.num_event_types
        self.backend = kwargs.get('backend', 'torch')
        self.kwargs = kwargs

    def build_input(self, source_dir, data_format, split):
        """Helper function to load and process dataset based on file format.

        Args:
            source_dir (str): Path to dataset directory.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary containing sequences of event times, types, and intervals.
        """

        if data_format == 'pkl':
            return self._build_input_from_pkl(source_dir, split)
        elif data_format == 'json':
            return self._build_input_from_json(source_dir, split)
        else:
            raise ValueError(f"Unsupported file format: {data_format}")

    def _build_input_from_pkl(self, source_dir, split):
        """Load and process data from a pickle file.

        Args:
            source_dir (str): Path to the pickle file.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary with processed event sequences.
        """
        data = load_pickle(source_dir)
        py_assert(data["dim_process"] == self.num_event_types,
                  ValueError, "Inconsistent dim_process in different splits.")

        source_data = data[split]
        return {
            'time_seqs': [[x["time_since_start"] for x in seq] for seq in source_data],
            'type_seqs': [[x["type_event"] for x in seq] for seq in source_data],
            'time_delta_seqs': [[x["time_since_last_event"] for x in seq] for seq in source_data]
        }

    def _build_input_from_json(self, source_dir, split):
        """Load and process data from a JSON file.

        Args:
            source_dir (str): Path to the JSON file or Hugging Face dataset name.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary with processed event sequences.
        """
        from datasets import load_dataset
        split_mapped = 'validation' if split == 'dev' else split
        if source_dir.endswith('.json'):
            data = load_dataset('json', data_files={split_mapped: source_dir}, split=split_mapped)
        elif source_dir.startswith('easytpp'):
            data = load_dataset(source_dir, split=split_mapped)
        else:
            raise ValueError("Unsupported source directory format for JSON.")

        py_assert(data['dim_process'][0] == self.num_event_types,
                  ValueError, "Inconsistent dim_process in different splits.")

        return {
            'time_seqs': data['time_since_start'],
            'type_seqs': data['type_event'],
            'time_delta_seqs': data['time_since_last_event']
        }

    def get_loader(self, split='train', **kwargs):
        """Get the corresponding data loader.

        Args:
            split (str, optional): denote the train, valid and test set. Defaults to 'train'.
            num_event_types (int, optional): num of event types in the data. Defaults to None.

        Raises:
            NotImplementedError: the input of 'num_event_types' is inconsistent with the data.

        Returns:
            EasyTPP.DataLoader: the data loader for tpp data.
        """
        data_dir = self.data_config.get_data_dir(split)
        data = self.build_input(data_dir, self.data_config.data_format, split)

        dataset = TPPDataset(data)
        tokenizer = EventTokenizer(self.data_config.data_specs)

        # Remove 'shuffle' from kwargs if it exists to avoid conflict
        shuffle = kwargs.pop('shuffle', self.kwargs.get('shuffle', False))

        loader = get_data_loader(dataset,
                                 self.backend,
                                 tokenizer,
                                 batch_size=self.kwargs['batch_size'],
                                 shuffle=shuffle,
                                 **kwargs)

        return loader

    def train_loader(self, **kwargs):
        """Return the train loader

        Returns:
            EasyTPP.DataLoader: data loader for train set.
        """
        return self.get_loader('train', **kwargs)

    def valid_loader(self, **kwargs):
        """Return the valid loader

        Returns:
            EasyTPP.DataLoader: data loader for valid set.
        """
        return self.get_loader('dev', **kwargs)

    def test_loader(self, **kwargs):
        """Return the test loader

        Returns:
            EasyTPP.DataLoader: data loader for test set.
        """
        # for test set, we do not shuffle
        kwargs['shuffle'] = False
        return self.get_loader('test', **kwargs)

    def get_statistics(self, split='train'):
        """Get basic statistics about the dataset.

        Args:
            split (str): Dataset split, e.g., 'train', 'dev', 'test'. Default is 'train'.

        Returns:
            dict: Dictionary containing statistics about the dataset.
        """
        data_dir = self.data_config.get_data_dir(split)
        data = self.build_input(data_dir, self.data_config.data_format, split)

        num_sequences = len(data['time_seqs'])
        sequence_lengths = [len(seq) for seq in data['time_seqs']]
        avg_sequence_length = sum(sequence_lengths) / num_sequences
        all_event_types = [event for seq in data['type_seqs'] for event in seq]
        event_type_counts = Counter(all_event_types)

        # Calculate time_delta_seqs statistics
        all_time_deltas = [delta for seq in data['time_delta_seqs'] for delta in seq]
        mean_time_delta = np.mean(all_time_deltas) if all_time_deltas else 0
        min_time_delta = np.min(all_time_deltas) if all_time_deltas else 0
        max_time_delta = np.max(all_time_deltas) if all_time_deltas else 0

        stats = {
            "num_sequences": num_sequences,
            "avg_sequence_length": avg_sequence_length,
            "event_type_distribution": dict(event_type_counts),
            "max_sequence_length": max(sequence_lengths),
            "min_sequence_length": min(sequence_lengths),
            "mean_time_delta": mean_time_delta,
            "min_time_delta": min_time_delta,
            "max_time_delta": max_time_delta
        }

        return stats

    def plot_event_type_distribution(self, split='train'):
        """Plot the distribution of event types in the dataset.

        Args:
            split (str): Dataset split, e.g., 'train', 'dev', 'test'. Default is 'train'.
        """
        stats = self.get_statistics(split)
        event_type_distribution = stats['event_type_distribution']

        plt.figure(figsize=(8, 6))
        plt.bar(event_type_distribution.keys(), event_type_distribution.values(), color='skyblue')
        plt.xlabel('Event Types')
        plt.ylabel('Frequency')
        plt.title(f'Event Type Distribution ({split} set)')
        plt.show()

    def plot_event_delta_times_distribution(self, split='train'):
        """Plot the distribution of event delta times in the dataset.

        Args:
            split (str): Dataset split, e.g., 'train', 'dev', 'test'. Default is 'train'.
        """
        data_dir = self.data_config.get_data_dir(split)
        data = self.build_input(data_dir, self.data_config.data_format, split)

        # Flatten the time_delta_seqs to get all delta times
        all_time_deltas = [delta for seq in data['time_delta_seqs'] for delta in seq]

        plt.figure(figsize=(10, 6))
        plt.hist(all_time_deltas, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Event Delta Times')
        plt.ylabel('Frequency')
        plt.title(f'Event Delta Times Distribution ({split} set)')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    def plot_sequence_length_distribution(self, split='train'):
        """Plot the distribution of sequence lengths in the dataset.

        Args:
            split (str): Dataset split, e.g., 'train', 'dev', 'test'. Default is 'train'.
        """
        data_dir = self.data_config.get_data_dir(split)
        data = self.build_input(data_dir, self.data_config.data_format, split)
        sequence_lengths = [len(seq) for seq in data['time_seqs']]

        plt.figure(figsize=(8, 6))
        plt.hist(sequence_lengths, bins=10, color='salmon', edgecolor='black')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.title(f'Sequence Length Distribution ({split} set)')
        plt.show()
