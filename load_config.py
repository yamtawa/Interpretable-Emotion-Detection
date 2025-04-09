import os
import yaml

def load_config(config_name='config'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'configs', f'{config_name}.yaml')
    # config_path = os.path.join(os.getcwd(), 'configs', fr'{config_name}.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_label_map(dataset_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    label_map_path = os.path.join(base_dir, 'configs', f'labels_mapping.yaml')
    with open(label_map_path, 'r') as file:
        label_map = yaml.safe_load(file)
    return label_map[dataset_name]


