import argparse
import os
import yaml
from os.path import dirname, abspath
import ruamel
import ruamel.yaml

class Parser:
    def __init__(self):
        self.L = None
        self.loc_T = None
        self.acc_fs = None
        self.threshold = None
        self.acc_length = None
        self.loc_length = None
        self.acc_stride = None
        self.loc_stride = None
        self.bag_size = None
        self.bag_step = None
        self.acc_form = None
        self.combine = None
        self.f_interp = None
        self.log_power = None
        self.stft_duration = None
        self.stft_overlap = None
        self.sync_thres = None
        self.loc_features = None
        self.architecture = None
        self.postprocess = None
        self.acc_signals = None
        self.loc_signals = None
        self.HR_sync_thres = None

        self.parser = argparse.ArgumentParser(
            description="preprocessing and inference parameters"
        )

    def __call__(self, *args, **kwargs):
        project_root = dirname(abspath(__file__))
        config_path = os.path.join(project_root, 'config.yaml')

        self.parser.add_argument(
            '--config',
            default=config_path,
            help='config file location'
        )

        self.parser.add_argument(
            '--acceleration',
            default=dict(),
            type=dict,
            help='acceleration preprocessing arguments'
        )

        self.parser.add_argument(
            '--location',
            default=dict(),
            type=dict,
            help='location preprocessing arguments'
        )

        self.parser.add_argument(
            '--HR',
            default=dict(),
            type=dict,
            help='heart rate preprocessing arguments'
        )

        self.parser.add_argument(
            '--inference',
            default=dict(),
            type=dict,
            help='inference arguments'
        )

    def get_args(self):
        self.__call__()
        args = self.parser.parse_args(args = [])
        config_file = args.config

        assert config_file is not None

        with open(config_file, 'r') as cf:
            deafault_args = yaml.load(cf, Loader=yaml.FullLoader)

        keys = vars(args).keys()

        for default_key in deafault_args.keys():
            if default_key not in keys:
                print('WRONG ARG: {}'.format(default_key))
                assert (default_key in keys)

        self.parser.set_defaults(**deafault_args)
        args = self.parser.parse_args(args = [])

        self.acc_fs = args.acceleration['fs']
        self.loc_T = args.location['T']
        self.threshold = args.acceleration['threshold']
        self.loc_signals = args.location['signals']
        self.acc_length = args.acceleration['length']
        self.loc_length = args.location['length']
        self.acc_stride = args.acceleration['stride']
        self.loc_stride = args.location['stride']
        self.bag_size = args.acceleration['bag_size']
        self.bag_step = args.acceleration['bag_step']
        self.acc_form = args.acceleration['form']
        self.combine = args.acceleration['combine']
        self.f_interp = args.acceleration['f_interp']
        self.log_power = args.acceleration['log_power']
        self.stft_duration = args.acceleration['stft_duration']
        self.stft_overlap = args.acceleration['stft_overlap']
        self.sync_thres = args.location['sync_thres']
        self.loc_features = args.location['features']
        self.architecture = args.inference['architecture']
        self.postprocess = args.inference['postprocess']
        self.acc_signals = args.acceleration['signals']
        self.L = args.inference['embedding_size']
        self.HR_sync_thres = args.HR['sync_thres']



