import unittest

import numpy as np
import torch
import os
import sys

# Get the directory of the current file
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_file_path)))

from easy_tpp.model import TorchNHP as NHP
from easy_tpp.preprocess.dataset import get_data_loader
from easy_tpp.config_factory import DataSpecConfig, ModelConfig
from easy_tpp.utils import load_json
from easy_tpp.preprocess.dataset import TPPDataset, EventTokenizer


class TestNeuralHawkesProcess(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        # Assuming the data is already generated and saved in 'synthetic_hf_data.json'
        self.data_file = 'synthetic_data.json'
        self.batch_size = 4
        self.input_data = self._make_json_2_dict(self.data_file)
        self.dataset = TPPDataset(self.input_data)

        config = DataSpecConfig.parse_from_yaml_config({'num_event_types': 3,
                                                        'batch_size': self.batch_size,
                                                        'pad_token_id': 3})

        self.tokenizer = EventTokenizer(config)

        self.data_loader = get_data_loader(self.dataset, 'torch', self.tokenizer, batch_size=self.batch_size)

        model_config = ModelConfig.parse_from_yaml_config({'hidden_size': 32,
                                                           'loss_integral_num_sample_per_step': 20,
                                                           'num_event_types': 3,
                                                           'num_event_types_pad': 4,
                                                           'event_pad_index': 3})
        self.model = NHP(model_config)

    def _make_json_2_dict(self, json_dir):
        json_data = load_json(json_dir)
        res = dict()
        res['time_seqs'] = [x['time_since_start'] for x in json_data]
        res['time_delta_seqs'] = ([np.array(x['time_since_last_event'], dtype=np.float32) for x in json_data])
        res['type_seqs'] = [x['type_event'] for x in json_data]
        return res

    def test_model_initialization(self):
        """Test if the model is initialized correctly."""
        self.assertIsInstance(self.model, NHP)
        self.assertEqual(self.model.hidden_size, 32)

    def test_forward_pass(self):
        """Test the forward pass of the model."""
        batch = next(iter(self.data_loader)).values()
        output = self.model(batch)
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertIsInstance(output[1], torch.Tensor)

    def test_loss_computation(self):
        """Test if the model computes loss correctly."""
        batch = next(iter(self.data_loader)).values()
        loss = self.model.loglike_loss(batch)
        self.assertGreater(loss[0].item(), 0)  # Loss should be positive

    def test_backward_pass(self):
        """Test if the model can perform a backward pass."""
        batch = next(iter(self.data_loader)).values()
        loss = self.model.loglike_loss(batch)
        loss[0].backward()
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)  # Ensure gradients are computed

    def test_training_step(self):
        """Test a single training step."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for batch in self.data_loader:
            optimizer.zero_grad()
            loss = self.model.loglike_loss(batch.values())
            loss[0].backward()
            optimizer.step()
            self.assertIsNotNone(loss[0])  # Ensure loss is computed
            break  # Only run one step for the test


if __name__ == '__main__':
    unittest.main()
