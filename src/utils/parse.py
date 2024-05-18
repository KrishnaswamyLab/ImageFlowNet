import ast
import os
from glob import glob

from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log


def parse_settings(config: AttributeHashmap, segmentor: bool = False,
                   log_settings: bool = True, run_count: int = None):
    # fix typing issues
    for key in ['learning_rate', 'ode_tol']:
        if key in config.keys():
            config[key] = float(config[key])

    # fix path issues
    ROOT = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
    for key in config.keys():
        if type(config[key]) == str and '$ROOT' in config[key]:
            config[key] = config[key].replace('$ROOT', ROOT)

    if segmentor:
        config.save_folder = os.path.dirname(config.segmentor_ckpt) + '/'
        os.makedirs(config.save_folder, exist_ok=True)
        config.model_save_path = config.segmentor_ckpt

    else:
        setting_str = '%s_%s_%ssmoothness-%.3f_latent-%.3f_contrastive-%.3f_invariance-%.3f_seed_%s' % (
            config.dataset_name,
            config.model,
            'NoL2_' if config.no_l2 else '',
            config.coeff_smoothness,
            config.coeff_latent,
            config.coeff_contrastive,
            config.coeff_invariance,
            config.random_seed,
        )

        output_save_path = '%s/%s' % (config.output_save_folder, setting_str)

        # Initialize save folder.
        if run_count is None:
            existing_runs = glob(output_save_path + '/run_*/')
            if len(existing_runs) > 0:
                run_counts = [int(item.split('/')[-2].split('run_')[1]) for item in existing_runs]
                run_count = max(run_counts) + 1
            else:
                run_count = 1

        config.save_folder = '%s/run_%d/' % (output_save_path, run_count)
        config.model_save_path = config.save_folder + setting_str + '.pty'

    # Initialize log file.
    config.log_dir = config.save_folder + 'log.txt'
    if log_settings:
        log_str = 'Config: \n'
        for key in config.keys():
            log_str += '%s: %s\n' % (key, config[key])
        log_str += '\nTraining History:'
        log(log_str, filepath=config.log_dir, to_console=True)

    return config
