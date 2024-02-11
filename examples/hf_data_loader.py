from datasets import load_dataset

def load_data_from_hf(hf_dir=None, local_dir=None):
    if hf_dir:
        ds = load_dataset(hf_dir)
    else:
        ds = load_dataset('json', data_files=local_dir)
    print(ds)
    print('dim process: ' + str(ds['validation'].data['dim_process'][0].as_py()))
    print('num seqs: ' + str(ds['validation'].data['num_seqs'][0].as_py()))
    print('avg seq len: ' + str(ds['validation'].data['avg_seq_len'][0].as_py()))
    print('min seq len: ' + str(ds['validation'].data['min_seq_len'][0].as_py()))
    print('max seq len: ' + str(ds['validation'].data['max_seq_len'][0].as_py()))
    return


if __name__ == '__main__':
    load_data_from_hf(hf_dir=None, local_dir={'validation':'dev.json'})