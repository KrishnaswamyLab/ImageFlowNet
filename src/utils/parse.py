import ast
import os

from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log


def parse_settings(config: AttributeHashmap, log_settings: bool = True):
    # fix typing issues
    for key in ['learning_rate', 'ode_tol']:
        if key in config.keys():
            config[key] = float(config[key])
    for key in ['target_dim']:
        if key in config.keys():
            config[key] = ast.literal_eval(config[key])

    # fix path issues
    ROOT = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
    for key in config.keys():
        if type(config[key]) == str and '$ROOT' in config[key]:
            config[key] = config[key].replace('$ROOT', ROOT)

    # Initialize log file.
    config.log_dir = config.log_folder + '/' + \
        os.path.basename(
            config.config_file_name).replace('.yaml', '') + '_log.txt'
    if log_settings:
        log_str = 'Config: \n'
        for key in config.keys():
            log_str += '%s: %s\n' % (key, config[key])
        log_str += '\nTraining History:'
        log(log_str, filepath=config.log_dir, to_console=True)
    return config
