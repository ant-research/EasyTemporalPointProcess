import unittest

from easy_tpp.config_factory import DataSpecConfig
from easy_tpp.utils import load_json
from easy_tpp.preprocess.dataset import TPPDataset, EventTokenizer, get_data_loader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
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

    def _make_json_2_dict(self, json_dir):
        json_data = load_json(json_dir)
        res = dict()
        res['time_seqs'] = [x['time_since_start'] for x in json_data]
        res['time_delta_seqs'] = [x['time_since_last_event'] for x in json_data]
        res['type_seqs'] = [x['type_event'] for x in json_data]
        return res

    def test_data_loading(self):
        """Test if data is loaded correctly."""
        self.assertIsNotNone(self.input_data)
        self.assertIsInstance(self.input_data, dict)
        self.assertGreater(len(self.input_data), 0)

    def test_batch_generation(self):
        """Test if batches are generated correctly."""
        self.assertGreater(len(self.data_loader), 0)
        for batch in self.data_loader:
            self.assertLessEqual(batch['time_seqs'].shape[0], self.batch_size)
            self.assertIn('time_seqs', batch)
            self.assertIn('time_delta_seqs', batch)
            self.assertIn('type_seqs', batch)

    def test_batch_content(self):
        """Test if batch content is correct."""
        for batch in self.data_loader:
            self.assertEqual(len(batch['time_seqs']), len(batch['time_delta_seqs']))
            self.assertEqual(len(batch['time_seqs']), len(batch['type_seqs']))


if __name__ == '__main__':
    unittest.main()
