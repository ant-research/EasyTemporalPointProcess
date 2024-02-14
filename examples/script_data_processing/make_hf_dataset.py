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

    dim_process = int(data_pkl['dim_process'])

    data_json = []
    for idx, seq in enumerate(data_pkl[split]):
        seq_len = len(seq)
        time_since_start, time_since_last_event, type_event = [], [], []
        for idx_event, event in enumerate(data_pkl[split][idx]):
            if idx_event == 0 and event['time_since_start'] > 0:
                start_timestamp = event['time_since_start']
                event['time_since_last_event'] -= start_timestamp if event[
                                                                         'time_since_last_event'] == start_timestamp else \
                    event['time_since_last_event']
                event['time_since_start'] -= start_timestamp
            event = make_json_serializable(event)
            time_since_start.append(event['time_since_start'])
            time_since_last_event.append(event['time_since_last_event'])
            type_event.append(event['type_event'])

        temp_dict = {'dim_process': dim_process,
                     'seq_idx': idx,
                     'seq_len': seq_len,
                     'time_since_start': time_since_start,
                     'time_since_last_event': time_since_last_event,
                     'type_event': type_event}
        data_json.append(temp_dict)

    with open(target_dir, "w") as outfile:
        json.dump(data_json, outfile)

    return


if __name__ == '__main__':
    make_hf_dataset('../data/taxi/test.pkl', 'test.json', split='test')
