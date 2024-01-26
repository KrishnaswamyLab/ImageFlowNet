import ast
import os
from glob import glob

from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log


def parse_settings(config: AttributeHashmap, log_settings: bool = True, run_count: int = None):
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

    # Initialize save folder.
    if run_count is None:
        existing_runs = glob(config.output_save_path + '/run_*/')
        if len(existing_runs) > 0:
            run_counts = [int(item.split('/')[-2].split('run_')[1]) for item in existing_runs]
            run_count = max(run_counts) + 1
        else:
            run_count = 1

    config.save_folder = '%s/run_%d/' % (config.output_save_path, run_count)

    # Initialize log file.
    config.log_dir = config.save_folder + 'log.txt'
    if log_settings:
        log_str = 'Config: \n'
        for key in config.keys():
            log_str += '%s: %s\n' % (key, config[key])
        log_str += '\nTraining History:'
        log(log_str, filepath=config.log_dir, to_console=True)

    config.model_save_path = \
        os.path.dirname(config.model_save_path) + '/run_%d/' % run_count + os.path.basename(config.model_save_path)
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

    return config
