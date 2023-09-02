import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# data source: https://earthquake.usgs.gov/earthquakes/search/

def event_type_map(mag):
    if mag < 2.75:
        return 0
    elif mag < 3.0:
        return 1
    elif mag < 3.5:
        return 2
    elif mag < 4.0:
        return 3
    elif mag < 4.5:
        return 4
    elif mag < 5.0:
        return 5
    else:
        return 6


def clean_csv(source_dir):
    df = pd.read_csv(source_dir, header=0)

    df.drop_duplicates(inplace=True)

    df.sort_values(by=['time'], inplace=True)
    print(len(df))
    df = df[['time', 'mag']]
    df['event_type'] = df['mag'].apply(lambda x: event_type_map(x))

    df.to_csv('earthquake.csv', index=False, header=True)
    return


def make_seq(df):
    seq = []
    df['time_diff'] = df['event_time'].diff()
    df.index = np.arange(len(df))
    for index, row in df.iterrows():
        if index == 0:
            event_dict = {"time_since_last_event": 0.0,
                          "time_since_start": 0.0,
                          "type_event": row['event_type']
                          }
            start_event_time = row['event_time']
        else:
            event_dict = {"time_since_last_event": row['time_diff'],
                          "time_since_start": row['event_time'] - start_event_time,
                          "type_event": row['event_type']
                          }
        seq.append(event_dict)

    return seq


def make_pkl(target_dir, dim_process, split, seqs):
    with open(target_dir, "wb") as f_out:
        pickle.dump(
            {
                "dim_process": dim_process,
                split: seqs
            }, f_out
        )
    return


def make_dataset(source_dir):
    df = pd.read_csv(source_dir, header=0)
    df['time'] = pd.to_datetime(df['time'])

    norm_const = 10000
    df['event_time'] = df['time'].apply(lambda x: datetime.timestamp(x)) / norm_const
    seq_len = np.random.randint(15, 19, 4300)
    print(np.sum(seq_len))

    seq_start_idx = [0] + list(np.cumsum(seq_len)[:-1] - 1)
    seq_end_idx = np.cumsum(seq_len) - 1

    total_seq = [make_seq(df.iloc[start_idx:end_idx, :]) for (start_idx, end_idx) in
                 zip(seq_start_idx, seq_end_idx)]

    print(len(total_seq))
    make_pkl('train.pkl', 7, 'train', total_seq[:3000])
    print(np.sum(seq_len[:3000]))
    make_pkl('dev.pkl', 7, 'dev', total_seq[3000:3400])
    print(np.sum(seq_len[3000:3400]))
    make_pkl('test.pkl', 7, 'test', total_seq[3400:])
    print(np.sum(seq_len[3400:]))

    # 70794
    # 4300
    # 49364
    # 6612
    # 14818

    return


if __name__ == '__main__':
    # clean_csv()
    make_dataset('earthquake.csv')
