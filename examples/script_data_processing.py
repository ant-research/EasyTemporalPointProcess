import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# source data: https://tianchi.aliyun.com/dataset/dataDetail?dataId=649

def check_dominate_event_type(event_type_seq, threshold=0.7):
    event_type = np.unique(event_type_seq)
    total_len = len(event_type_seq)
    type_ratio = [len(event_type_seq[event_type_seq == event_type_i]) / total_len for event_type_i in event_type]

    return True if max(type_ratio) > threshold else False


def cate_map(cate_id, cate_event_map_df):
    res = cate_event_map_df[cate_event_map_df['cate'] == cate_id]['event_id'].to_list()[0]
    return res


def read_data_step_3(source_dir, cate_dir, target_dir):
    train_df = pd.read_csv(source_dir, header=0)

    cate_event_map_df = pd.read_csv(cate_dir, header=0)

    train_df['event_type'] = train_df['cate_id'].apply(lambda x: cate_map(x, cate_event_map_df))
    print(train_df['event_type'].value_counts(normalize=True))
    unique_user_id = np.unique(train_df['user_id'])

    for idx, user_id in enumerate(unique_user_id):
        user_df = train_df[train_df['user_id'] == user_id]
        prev_time = user_df.iloc[0, 4]
        event_dtime = user_df['event_dtime'].values
        event_time = user_df['event_time'].values
        event_dtime[0] = 0.0

        for i in range(1, len(event_time)):
            if event_dtime[i] > 3.0:
                rand_dt = np.random.random() + 0.1
                event_time[i] = prev_time + rand_dt
                event_dtime[i] = rand_dt
            else:
                event_time[i] = event_time[i - 1] + event_dtime[i]
            prev_time = event_time[i]

        user_df['event_dtime'] = event_dtime
        user_df['event_time'] = event_time

        print(min(event_dtime[1:]), max(event_dtime))

        assert abs(np.mean(user_df['event_time'].diff().values[1:]) - np.mean(event_dtime[1:])) < 0.0001

    train_df.to_csv(target_dir)
    return


def read_data_step_2(source_dir):
    train_df = pd.read_csv(source_dir, header=None)
    train_df.columns = ['user_id', 'item_id', 'cate_id', 'event_type_raw', 'event_time']
    count = train_df['cate_id'].value_counts(normalize=True)
    pd.DataFrame(count).to_csv('taobao_map.csv', header=True)

    return


def read_data_step_1(source_dir, target_dir):
    train_df = pd.read_csv(source_dir, header=None)
    train_df.columns = ['user_id', 'item_id', 'cate_id', 'event_type_raw', 'event_time']
    train_df['event_time'] /= 10000
    unique_user_id = np.unique(train_df['user_id'])

    train_df = train_df[train_df['event_type_raw'] == 'pv']

    res = pd.DataFrame()
    total_seq = 0

    for idx, user_id in enumerate(unique_user_id):
        print(f'user {idx}')
        user_df = train_df[train_df['user_id'] == user_id]

        # drop consecutive duplicate on pv
        user_df = user_df.loc[user_df['cate_id'].shift() != user_df['cate_id']]
        user_df.fillna(0.0, inplace=True)

        user_df.sort_values(by=['event_time'], inplace=True)
        user_df['event_dtime'] = user_df['event_time'].diff()

        user_df.fillna(0.0, inplace=True)

        # drop dtime < 0.05
        user_df = user_df[user_df['event_dtime'] > 0.1]

        if len(user_df) < 40:
            print('user seq is too short, skip it')
            continue

        total_seq += 1
        print(f'{total_seq} users have been recorded')
        res = pd.concat([res, user_df])
        if total_seq > 2000:
            break

    res.to_csv(target_dir, header=True, index=False)

    return


def save_data(source_dir):
    df = pd.read_csv(source_dir, header=0)
    unique_user_id = np.unique(df['user_id'])
    res = []
    print(np.unique(df['event_type']))
    for idx, user_id in enumerate(unique_user_id):
        print(f'user {idx}')
        user_seq = []
        user_df = df[df['user_id'] == user_id]
        length = 0
        for idx_row, row in user_df.iterrows():
            event_dtime = 0 if length == 0 else row['event_dtime']
            user_seq.append({"time_since_last_event": event_dtime,
                             "time_since_start": row['event_time'],
                             "type_event": row['event_type']
                             })
            length += 1

        res.append(user_seq)

    with open('../data/taobao/train.pkl', "wb") as f_out:
        pickle.dump(
            {
                "dim_process": 17,
                'train': res[:1300]
            }, f_out
        )

    with open('../data/taobao/dev.pkl', "wb") as f_out:
        pickle.dump(
            {
                "dim_process": 17,
                'dev': res[1300:1500]
            }, f_out
        )

    with open('../data/taobao/test.pkl', "wb") as f_out:
        pickle.dump(
            {
                "dim_process": 17,
                'test': res[1500:]
            }, f_out
        )

    return
