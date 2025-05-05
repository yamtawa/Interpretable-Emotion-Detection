import os
import yaml

def load_config(config_name='config'):
    config_path = os.path.join(os.getcwd(), 'configs', fr'{config_name}.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


