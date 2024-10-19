from easy_tpp.utils.gen_utils import generate_and_save_json

if __name__ == "__main__":
    generate_and_save_json(n_nodes=3,
                           end_time=100,
                           baseline=1,
                           adjacency=0.5,
                           decay=0.1,
                           max_seq_len=40,
                           target_file='synthetic_data.json')
