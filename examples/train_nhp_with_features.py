import random
from typing import Optional, Union, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from easy_tpp.config_factory import DataSpecConfig, Config
from easy_tpp.model import TorchNHP as NHP
from easy_tpp.preprocess import TPPDataset, EventTokenizer
from easy_tpp.preprocess.data_collator import TPPDataCollator
from easy_tpp.preprocess.event_tokenizer import BatchEncoding
from easy_tpp.utils import PaddingStrategy


def make_raw_data():
    data = [
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 0, 'loan_amt': 10}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1, 'loan_amt': 10}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1, 'loan_amt': 20}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1, 'loan_amt': 20}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1, 'loan_amt': 20}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1, 'loan_amt': 30}],
    ]
    for i, j in enumerate([2, 5, 3, 2, 4, 2]):
        start_time = 0
        for k in range(j):
            delta_t = random.random()
            start_time += delta_t
            data[i].append({"time_since_last_event": delta_t,
                            "time_since_start": start_time,
                            "type_event": random.randint(0, 10),
                            'loan_amt': random.randint(10, 30)})

    return data


class TPPDatasetV2(TPPDataset):
    def __init__(self, data):
        super(TPPDatasetV2, self).__init__(data)
        self.loan_amt_seqs = self.data_dict['loan_amt_seqs']

    def __getitem__(self, idx):
        """

        Args:
            idx: iteration index

        Returns:
            dict: a dict of time_seqs, time_delta_seqs and type_seqs element

        """
        return dict({'time_seqs': self.time_seqs[idx], 'time_delta_seqs': self.time_delta_seqs[idx],
                     'type_seqs': self.type_seqs[idx], 'loan_amt_seqs': self.loan_amt_seqs[idx]})


class EventTokenizerV2(EventTokenizer):
    def __init__(self, config):
        super(EventTokenizerV2, self).__init__(config)
        self.model_input_names.append('loan_amt_seqs')

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, Any], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        # check whether we need to pad it
        is_all_seq_equal_max_length = [len(seq) == max_length for seq in required_input]
        is_all_seq_equal_max_length = np.prod(is_all_seq_equal_max_length)
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and ~is_all_seq_equal_max_length

        batch_output = dict()

        if needs_to_be_padded:
            # time seqs
            batch_output[self.model_input_names[0]] = self.make_pad_sequence(encoded_inputs[self.model_input_names[0]],
                                                                             self.pad_token_id,
                                                                             padding_side=self.padding_side,
                                                                             max_len=max_length)
            # time_delta seqs
            batch_output[self.model_input_names[1]] = self.make_pad_sequence(encoded_inputs[self.model_input_names[1]],
                                                                             self.pad_token_id,
                                                                             padding_side=self.padding_side,
                                                                             max_len=max_length)
            # type_seqs
            batch_output[self.model_input_names[2]] = self.make_pad_sequence(encoded_inputs[self.model_input_names[2]],
                                                                             self.pad_token_id,
                                                                             padding_side=self.padding_side,
                                                                             max_len=max_length,
                                                                             dtype=np.int32)

        else:
            batch_output = encoded_inputs

        # non_pad_mask
        # we must use type seqs to check the mask, because the pad_token_id maybe one of valid values in
        # time seqs
        seq_pad_mask = batch_output[self.model_input_names[2]] == self.pad_token_id
        batch_output[self.model_input_names[3]] = ~ seq_pad_mask

        if return_attention_mask:
            # attention_mask
            batch_output[self.model_input_names[4]] = self.make_attn_mask_for_pad_sequence(
                batch_output[self.model_input_names[2]],
                self.pad_token_id)
        else:
            batch_output[self.model_input_names[4]] = []

        # type_mask
        batch_output[self.model_input_names[5]] = self.make_type_mask_for_pad_sequence(
            batch_output[self.model_input_names[2]])

        # loan_amt_seqs
        batch_output[self.model_input_names[6]] = self.make_pad_sequence(encoded_inputs[self.model_input_names[-1]],
                                                                         self.pad_token_id,
                                                                         padding_side=self.padding_side,
                                                                         max_len=max_length)

        return batch_output


def make_data_loader():
    source_data = make_raw_data()

    time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
    type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
    time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]
    loan_amt_seqs = [[x["loan_amt"] for x in seq] for seq in source_data]

    input_data = {'time_seqs': time_seqs,
                  'type_seqs': type_seqs,
                  'time_delta_seqs': time_delta_seqs,
                  'loan_amt_seqs': loan_amt_seqs}

    config = DataSpecConfig.parse_from_yaml_config({'num_event_types': 11, 'batch_size': 1,
                                                    'pad_token_id': 11})

    dataset = TPPDatasetV2(input_data)

    tokenizer = EventTokenizerV2(config)

    padding = True if tokenizer.padding_strategy is None else tokenizer.padding_strategy
    truncation = False if tokenizer.truncation_strategy is None else tokenizer.truncation_strategy

    data_collator = TPPDataCollator(tokenizer=tokenizer,
                                    return_tensors='pt',
                                    max_length=tokenizer.model_max_length,
                                    padding=padding,
                                    truncation=truncation)

    data_loader = DataLoader(dataset, collate_fn=data_collator, batch_size=1)

    return data_loader


class NHPV2(NHP):
    def __init__(self, model_config):
        super(NHPV2, self).__init__(model_config)

        self.layer_loan_amt = nn.Linear(1, model_config.hidden_size)

        self.layer_merge = nn.Linear(model_config.hidden_size * 2, model_config.hidden_size)

    def forward(self, batch, **kwargs):
        """Call the model.

        Args:
            batch (tuple, list): batch input.

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens;
                  stacked decay states,  [batch_size, max_seq_length, 4, hidden_dim], states right after
                  the event happens.
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _, type_mask, loan_amt_seq = batch

        all_hiddens = []
        all_outputs = []
        all_cells = []
        all_cell_bars = []
        all_decays = []

        max_steps = kwargs.get('max_steps', None)

        max_decay_time = kwargs.get('max_decay_time', 5.0)

        # last event has no time label
        max_seq_length = max_steps if max_steps is not None else event_seq.size(1) - 1

        batch_size = len(event_seq)
        h_t, c_t, c_bar_i = self.init_state(batch_size)

        # if only one event, then we dont decay
        if max_seq_length == 1:
            types_sub_batch = event_seq[:, 0]
            x_t = self.layer_type_emb(types_sub_batch)

            # i add loan emb here
            loan_t = self.layer_loan_amt(loan_amt_seq[:, 0])
            x_t = self.layer_merge(torch.cat(x_t, loan_t))

            cell_i, c_bar_i, decay_i, output_i = \
                self.rnn_cell(x_t, h_t, c_t, c_bar_i)

            # Append all output
            all_outputs.append(output_i)
            all_decays.append(decay_i)
            all_cells.append(cell_i)
            all_cell_bars.append(c_bar_i)
            all_hiddens.append(h_t)
        else:
            # Loop over all events
            for i in range(max_seq_length):
                if i == event_seq.size(1) - 1:
                    dt = torch.ones_like(time_delta_seq[:, i]) * max_decay_time
                else:
                    dt = time_delta_seq[:, i + 1]  # need to carefully check here
                types_sub_batch = event_seq[:, i]
                x_t = self.layer_type_emb(types_sub_batch)

                # i add loan emb here
                loan_t = self.layer_loan_amt(loan_amt_seq[:, i:i+1])
                x_t = self.layer_merge(torch.cat([x_t, loan_t], dim=-1))

                # cell_i  (batch_size, process_dim)
                cell_i, c_bar_i, decay_i, output_i = \
                    self.rnn_cell(x_t, h_t, c_t, c_bar_i)

                # States decay - Equation (7) in the paper
                c_t, h_t = self.rnn_cell.decay(cell_i,
                                               c_bar_i,
                                               decay_i,
                                               output_i,
                                               dt[:, None])

                # Append all output
                all_outputs.append(output_i)
                all_decays.append(decay_i)
                all_cells.append(cell_i)
                all_cell_bars.append(c_bar_i)
                all_hiddens.append(h_t)

        # (batch_size, max_seq_length, hidden_dim)
        cell_stack = torch.stack(all_cells, dim=1)
        cell_bar_stack = torch.stack(all_cell_bars, dim=1)
        decay_stack = torch.stack(all_decays, dim=1)
        output_stack = torch.stack(all_outputs, dim=1)

        # [batch_size, max_seq_length, hidden_dim]
        hiddens_stack = torch.stack(all_hiddens, dim=1)

        # [batch_size, max_seq_length, 4, hidden_dim]
        decay_states_stack = torch.stack((cell_stack,
                                          cell_bar_stack,
                                          decay_stack,
                                          output_stack),
                                         dim=2)

        return hiddens_stack, decay_states_stack

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _, type_mask, loan_amt_seq = batch

        hiddens_ti, decay_states = self.forward(batch)

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = hiddens_ti.size()

        # Lambda(t) right before each event time point
        # lambda_at_event - [batch_size, num_times=max_len-1, num_event_types]
        # Here we drop the last event because it has no delta_time label (can not decay)
        lambda_at_event = self.layer_intensity(hiddens_ti)

        # Compute the big lambda integral in Equation (8)
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length

        # interval_t_sample - [batch_size, num_times=max_len-1, num_mc_sample]
        # for every batch and every event point => do a sampling (num_mc_sampling)
        # the first dtime is zero, so we use time_delta_seq[:, 1:]
        interval_t_sample = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # [batch_size, num_times = max_len - 1, num_mc_sample, hidden_size]
        state_t_sample = self.compute_states_at_sample_times(decay_states, interval_t_sample)

        # [batch_size, num_times = max_len - 1, num_mc_sample, event_num]
        lambda_t_sample = self.layer_intensity(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # (num_samples, num_times)
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

def make_model():
    config = Config.build_from_yaml_file('configs/experiment_config.yaml', experiment_id='NHP_train')
    model_config = config.model_config

    # hack this
    model_config.num_event_types = 11
    model_config.num_event_types_pad = 12
    model_config.pad_token_id = 11

    model = NHPV2(model_config)

    return model


def main():
    data_loader = make_data_loader()

    model = make_model()

    num_epochs = 10

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(num_epochs):
        total_loss = 0
        total_num_event = 0
        for batch in data_loader:
            with torch.set_grad_enabled(True):
                batch_loss, batch_num_event = model.loglike_loss(batch = batch.values())

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            total_loss += batch_loss
            total_num_event += batch_num_event

        avg_loss = total_loss / total_num_event
        print(f'epochs {i}: loss {avg_loss}')

    return


if __name__ == '__main__':
    main()
