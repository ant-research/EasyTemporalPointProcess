import json

import numpy as np

from easy_tpp.utils import load_pickle


def make_json_serializable(input_dict):
    for k, v in input_dict.items():
        if isinstance(v, np.float32):
            input_dict[k] = float(v)
        elif isinstance(v, np.int32):
            input_dict[k] = int(v)

    return input_dict


def make_hf_dataset(source_dir, target_dir, split='test'):
    data_pkl = load_pickle(source_dir)

    data_pkl['dim_process'] = int(data_pkl['dim_process'])

    data_json = dict({'dim_process': int(data_pkl['dim_process'])})

    data_json['event_seqs'] = dict()

    seq_len = []
    for idx, seq in enumerate(data_pkl[split]):
        data_json['event_seqs'][f'seq_{idx}'] = dict()
        seq_len.append(len(seq))
        for idx_event, event in enumerate(data_pkl[split][idx]):
            if idx_event == 0:
                start_timestamp = event['time_since_start']
                event['time_since_last_event'] -= start_timestamp if event[
                                                                         'time_since_last_event'] == start_timestamp else \
                event['time_since_last_event']
            event['time_since_start'] -= start_timestamp
            data_json['event_seqs'][f'seq_{idx}'][f'event_{idx_event}'] = make_json_serializable(event)

    data_json['num_seqs'] = len(data_pkl[split])
    data_json['avg_seq_len'] = np.mean(seq_len)
    data_json['min_seq_len'] = min(seq_len)
    data_json['max_seq_len'] = max(seq_len)

    with open(target_dir, "w") as outfile:
        json.dump(data_json, outfile)

    return
