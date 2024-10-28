import os 
import sys
# Get the directory of the current file
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

from easy_tpp.config_factory import Config
from easy_tpp.preprocess.data_loader import TPPDataLoader


def main():
    config = Config.build_from_yaml_file('./config.yaml')
    tpp_loader = TPPDataLoader(config)
    stats = tpp_loader.get_statistics(split='train')
    print(stats)
    tpp_loader.plot_event_type_distribution()
    tpp_loader.plot_event_delta_times_distribution()    

if __name__ == '__main__':
    main()