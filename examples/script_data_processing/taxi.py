import pickle
import warnings

warnings.filterwarnings('ignore')

def read_data_step_1():

    def read_pkl(file_dir):
        res = []
        taxi = pickle.load(open(file_dir, "rb" ))
        count = 0
        for seq in taxi['seqs']:
            if len(seq) > 34:
                count += 1
                res.append(seq)
                # print(np.max(seq['time_since_last_event']))
        print(count)
        return res

    # from Mei et al 's paper on event imputation
    train_res = read_pkl('pilottaxi/big/train.pkl')
    dev_res = read_pkl('pilottaxi/big/dev.pkl')
    test_res = read_pkl('pilottaxi/big/test1.pkl')

    with open('../data/taxi/train.pkl', "wb") as f_out:
        pickle.dump(
            {
                "dim_process": 10,
                'train': train_res[:1500]
            }, f_out
        )

    with open('../data/taxi/dev.pkl', "wb") as f_out:
        pickle.dump(
            {
                "dim_process": 10,
                'dev': dev_res[:200]
            }, f_out
        )

    with open('../data/taxi/test.pkl', "wb") as f_out:
        pickle.dump(
            {
                "dim_process": 10,
                'test': test_res[:400]
            }, f_out
        )

    return

if __name__ == '__main__':
    read_data_step_1()