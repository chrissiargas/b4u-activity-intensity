from config_parser import Parser
import numpy as np
from scipy.signal import spectrogram
from scipy.interpolate import interp2d
from typing import *
from geopy.distance import great_circle


class temper:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.fs = self.conf.acc_fs

        self.pivot = self.conf.bag_size // 2
        self.shape = self.get_shape()

    def get_shape(self):
        if self.conf.combine == 'concat':
            self.channels = len(self.conf.acc_signals)
            return self.conf.bag_size, self.conf.acc_length, self.channels

    def __call__(self, instances: np.ndarray, channels: Dict, preprocessing: bool = False):
        if preprocessing:
            outputs = {}

        elif self.conf.combine == 'concat':
            outputs = np.zeros(self.shape)
            channel = 0

        if preprocessing or self.conf.combine == 'concat':
            for signal in self.conf.acc_signals:
                if signal in channels.keys():
                    key = channels[signal]
                    x = instances[:, :, key]

                    if preprocessing:
                        outputs[signal] = x

                    elif self.conf.combine == 'concat':
                        outputs[..., channel] = x
                        channel += 1

        if preprocessing:
            return outputs

        elif self.conf.combine == 'concat':
            return [outputs]


class specter:
    def __init__(self):
        self.channels = None
        self.conf = Parser()
        self.conf.get_args()

        self.fs = self.conf.acc_fs

        self.pivot = self.conf.bag_size // 2
        self.height, self.width = 48, 48
        self.out_size = (self.height, self.width)
        self.nperseg = self.conf.stft_duration * self.fs
        self.noverlap = self.conf.stft_overlap * self.fs
        self.shape = self.get_shape()
        self.temper = temper()

    def get_shape(self):
        if self.conf.combine == 'concat':
            self.channels = len(self.conf.acc_signals)
            return self.conf.bag_size, self.height, self.width, self.channels

    def resize(self, spectros, freq, time):
        n = spectros.shape[0]
        out_f, out_t = self.height, self.width
        out_spectrograms = np.zeros((n, out_f, out_t), dtype=np.float32)

        if self.conf.f_interp == 'log':
            log_f = np.log(freq + freq[1])
            log_f_normalized = (log_f - log_f[0]) / (log_f[-1] - log_f[0])
            f = out_f * log_f_normalized

        t_normalized = (time - time[0]) / (time[-1] - time[0])
        t = out_t * t_normalized

        f_i = np.arange(out_f)
        t_i = np.arange(out_t)

        for i, spectro in enumerate(spectros):
            spectrogram_fn = interp2d(t, f, spectro, copy=False)
            out_spectrograms[i, :, :] = spectrogram_fn(f_i, t_i)

        return out_spectrograms

    def get_spectrogram(self, x):
        f, t, spectro = spectrogram(x, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        spectro = self.resize(spectro, f, t)

        if self.conf.log_power:
            np.log(spectro + 1e-10, dtype=np.float32, out=spectro)

        return spectro

    def __call__(self, instances: np.ndarray, channels: Dict):
        instances = instances.astype(np.float32)

        instances = self.temper(instances, channels, preprocessing=True)
        channel = 0

        if self.conf.combine == 'concat':
            outputs = np.zeros(self.shape)
            for signal in instances.keys():
                x = instances[signal]
                spectro = self.get_spectrogram(x)
                outputs[..., channel] = spectro
                channel += 1

            outputs = [outputs]

        return outputs

class locformer:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.stat_features = ['min', 'max', 'mean', 'std']
        self.ar_features = ['moveability']

        self.shape = self.get_shape()
        self.NaN_value = 0

    def get_shape(self):
        self.channels = len(self.conf.loc_signals)

        window_shape = 1, self.conf.loc_length, self.channels

        in_stat_features = [feature for feature in self.conf.loc_features if feature in self.stat_features]
        self.n_features = self.channels * len(in_stat_features)

        in_ar_features = [feature for feature in self.conf.loc_features if feature in self.ar_features]
        self.n_features += len(in_ar_features)

        features_shape = 1, self.n_features

        return window_shape, features_shape

    def get_distance(self, lat, long, i):
        if np.isnan([lat[i], long[i], lat[i - 1], long[i - 1]]).any():
            return np.nan

        point1 = (lat[i - 1], long[i - 1])
        point2 = (lat[i], long[i])
        distance = great_circle(point1, point2).m
        return distance

    def __call__(self, input, channels: Dict):
        input = input.astype(np.float32)

        window = np.zeros((self.conf.loc_length, self.channels))
        channel = 0
        for feature in self.conf.loc_signals:
            if feature in channels.keys():
                key = channels[feature]
                timeseq = input[:, key]
                window[:, channel] = timeseq
                channel += 1

        n = 0
        features = np.zeros(self.n_features)
        for feature in self.conf.loc_features:
            if feature in self.stat_features:
                if feature == 'mean':
                    values = [np.mean(window[:, i]) for i in range(window.shape[1])]

                elif feature == 'std':
                    values = [np.std(window[:, i]) for i in range(window.shape[1])]

                elif feature == 'min':
                    values = [np.min(window[:, i]) for i in range(window.shape[1])]

                elif feature == 'max':
                    values = [np.max(window[:, i]) for i in range(window.shape[1])]

                features[n: n + len(values)] = values
                n += len(values)

            if feature in self.ar_features:
                if feature == 'moveability':
                    if np.isnan(input).any():
                        value = np.nan
                    else:
                        lat = input[:, channels['latitude']]
                        long = input[:, channels['longitude']]

                        start_point = (lat[0], long[0])
                        end_point = (lat[-1], long[-1])
                        total_distance = great_circle(start_point, end_point).m

                        distances = [self.get_distance(lat, long, i) for i in range(1, input.shape[0])]
                        sum_distance = sum(distances)

                        value = total_distance / (sum_distance + 1e-10)

                features[n] = value
                n += 1

        window = window[np.newaxis, :]
        features = features[np.newaxis, :]

        window = np.nan_to_num(window, nan=self.NaN_value)
        features = np.nan_to_num(features, nan=self.NaN_value)

        return window, features











