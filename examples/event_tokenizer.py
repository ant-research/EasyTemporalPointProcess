import random

from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.config_factory import DataSpecConfig

def make_raw_data():
    data = [
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 0}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1}],
    ]
    for i, j in enumerate([2, 5, 3]):
        start_time = 0
        for k in range(j):
            delta_t = random.random()
            start_time += delta_t
            data[i].append({"time_since_last_event": delta_t,
                            "time_since_start": start_time,
                            "type_event": random.randint(0, 10)
                            })

    return data


def main():
    source_data = make_raw_data()

    time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
    type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
    time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]

    input_data = {'time_seqs': time_seqs,
                  'type_seqs': type_seqs,
                  'time_delta_seqs': time_delta_seqs}

    config = DataSpecConfig.parse_from_yaml_config({'num_event_types': 11,  'pad_token_id': 11})

    tokenizer = EventTokenizer(config)

    output = tokenizer.pad(input_data, return_tensors='pt')

    print(output)


if __name__ == '__main__':
    main()
