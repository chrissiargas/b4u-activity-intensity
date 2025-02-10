import numpy as np
import pandas as pd
from typing import Optional

from config_parser import Parser
from typing import Tuple, Dict, List
from utils import acc_produce, loc_produce, acc_segment, loc_segment, bagging, acc_form
import tensorflow as tf
from transformers import temper, specter, locformer
from plots import plot_window
import matplotlib.pyplot as plt

class builder:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.get_transformers()

    def get_transformers(self):
        if self.conf.acc_form == 'spectrogram':
            self.acc_transformer = specter()

        if self.conf.acc_form == 'temporal':
            self.acc_transformer = temper()

        self.acc_shape = self.acc_transformer.shape

        self.loc_transformer = locformer()
        self.loc_shape, self.loc_features_shape = self.loc_transformer.shape

        self.input_shape = (self.acc_shape, self.loc_shape, self.loc_features_shape)
        self.input_type = (*[tf.float32 for _ in self.acc_shape], tf.float32, tf.float32)

        self.output_shape = 1
        self.output_type = tf.float32

    def split_instances(self, acc, i):
        instances = []
        full_segment = acc[i]

        for s in range(self.conf.bag_size):
            instances.append(full_segment[s * self.conf.acc_stride:
                                          s * self.conf.acc_stride + self.conf.acc_length])

        return np.array(instances)

    def make_bags(self, acc, loc, i, j):
        acc_instances = self.split_instances(acc, i)
        acc_instances = self.acc_transformer(acc_instances, self.acc_channels)

        if j == -1:
            loc_window = np.zeros((self.conf.loc_length, len(self.loc_channels)))
            loc_window[...] = np.nan
        else:
            loc_window = loc[j]

        loc_window, loc_features = self.loc_transformer(loc_window, self.loc_channels)

        return *acc_instances, loc_window, loc_features

    def to_batch(self, acc, loc):
        batch = []
        for i, j in enumerate(self.syncing):
            X = self.make_bags(acc, loc, i, j)
            batch.append(X)

        batch = [np.array([input[i] for input in batch]) for i in range(len(X))]
        return batch

    def __call__(self, data: Tuple[pd.DataFrame, pd.DataFrame], verbose: bool = False):
        acc, loc = data

        acc = acc_form(acc, threshold=self.conf.threshold)

        acc = acc_produce(acc, self.conf.acc_signals)
        loc = loc_produce(loc, self.conf.loc_signals)

        acc, acc_t, self.acc_channels = acc_segment(acc, self.conf.acc_length, self.conf.acc_stride, self.conf.bag_size, self.conf.bag_step)
        loc, loc_t, self.loc_channels = loc_segment(loc, self.conf.loc_length, self.conf.loc_stride)

        if verbose:
            plot_window(acc, acc_t, self.acc_channels, features=['acc_x', 'acc_y', 'acc_z', 'norm_xyz'],
                        subject='auth-test1-38', timestamp = '2025-02-03 09:37:59')
            plot_window(loc, loc_t, self.loc_channels, features=['velocity', 'acceleration'],
                        subject='auth-test1-38', timestamp = '2025-02-03 09:37:59')

        self.syncing = bagging(acc_t, loc_t, self.conf.sync_thres)
        batch = self.to_batch(acc, loc)

        return batch, acc_t














