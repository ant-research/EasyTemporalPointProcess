from datasets import load_dataset

def load_data_from_hf(hf_dir=None, local_dir=None):
    if hf_dir:
        ds = load_dataset(hf_dir)
    else:
        ds = load_dataset('json', data_files=local_dir)
    
    print("Dataset structure:")
    print(ds)
    
    # Print available features for validation split
    print("\nValidation split features:")
    print(ds['validation'].features)
    
    # Try to access metadata fields if they exist, otherwise show available data
    try:
        print('\ndim process: ' + str(ds['validation'].data['dim_process'][0].as_py()))
    except (KeyError, IndexError):
        print("dim_process field not found in dataset")
    
    try:
        print('num seqs: ' + str(ds['validation'].data['num_seqs'][0].as_py()))
    except (KeyError, IndexError):
        print("num_seqs field not found in dataset")
    
    try:
        print('avg seq len: ' + str(ds['validation'].data['avg_seq_len'][0].as_py()))
    except (KeyError, IndexError):
        print("avg_seq_len field not found in dataset")
    
    try:
        print('min seq len: ' + str(ds['validation'].data['min_seq_len'][0].as_py()))
    except (KeyError, IndexError):
        print("min_seq_len field not found in dataset")
    
    try:
        print('max seq len: ' + str(ds['validation'].data['max_seq_len'][0].as_py()))
    except (KeyError, IndexError):
        print("max_seq_len field not found in dataset")
    
    # Show actual data structure
    print("\nFirst few examples from validation split:")
    for i, example in enumerate(ds['validation']):
        if i < 3:  # Show first 3 examples
            print(f"Example {i}:")
            for key, value in example.items():
                if isinstance(value, list) and len(value) > 10:
                    print(f"  {key}: {value[:5]}... (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
        else:
            break
    
    return ds


if __name__ == '__main__':
    # in case one fails to load from hf directly
    # one can load the json data file locally
    # load_data_from_hf(hf_dir=None, local_dir={'validation':'dev.json'})
    load_data_from_hf(hf_dir='easytpp/taxi')