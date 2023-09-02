import datetime
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def make_datetime(year, month, day):
    try:
        date = datetime.datetime(int(year), int(month), int(day))
    except ValueError as e:
        if e.args[0] == 'day is out of range for month':
            date = datetime.datetime(int(year), int(month), int(day)-1)
    return datetime.datetime.timestamp(date) + 61851630000   # make sure the timestamp is positive


def clean_csv():
    source_dir = 'events.csv'

    df = pd.read_csv(source_dir, header=0)

    df = df[~df['event_date_year'].isna()]
    df = df[df['event_date_year'] > 0]
    df['event_date_month'].fillna(1, inplace=True)
    df['event_date_day'].fillna(1, inplace=True)
    df.drop_duplicates(inplace=True)
    norm_const = 1000000
    df['event_timestamp'] = df.apply(
        lambda x: make_datetime(x['event_date_year'], x['event_date_month'], x['event_date_day']),
        axis=1)/norm_const
    df.sort_values(by=['event_date_year', 'event_date_month', 'event_date_day'], inplace=True)
    df['event_type'] = [0] * len(df)

    df.to_csv('volcano.csv', index=False, header=True)
    return


def make_seq(df):
    seq = []
    df['time_diff'] = df['event_timestamp'].diff()
    df.index = np.arange(len(df))
    for index, row in df.iterrows():
        if index == 0:
            event_dict = {"time_since_last_event": 0.0,
                          "time_since_start": 0.0,
                          "type_event": row['event_type']
                          }
            start_event_time = row['event_timestamp']
        else:
            event_dict = {"time_since_last_event": row['time_diff'],
                          "time_since_start": row['event_timestamp'] - start_event_time,
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

    vols = np.unique(df['volcano_name'])
    total_seq = []
    for vol in vols:
        df_ = df[df['volcano_name'] == vol]
        df_.sort_values('event_timestamp', inplace=True)
        total_seq.append(make_seq(df_))


    print(len(total_seq))
    make_pkl('train.pkl', 1, 'train', total_seq[:400])
    count_seq(total_seq[:400])
    make_pkl('dev.pkl', 1, 'dev', total_seq[400:450])
    count_seq(total_seq[400:450])
    make_pkl('test.pkl', 1, 'test', total_seq[450:])
    count_seq(total_seq[450:])


    return


def count_seq(seqs):
    total_len = [len(seq) for seq in seqs]
    print(np.mean(total_len))
    print(np.sum(total_len))

    return

if __name__ == '__main__':
    # clean_csv()
    make_dataset('volcano.csv')
