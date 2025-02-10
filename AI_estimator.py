
import os

import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib

from config_parser import Parser
from preprocessing import builder
from architectures import get_fusion_MIL
import time
from typing import Dict, Tuple

get_dist = np.vectorize(lambda x, y: (pd.to_datetime(x) - pd.to_datetime(y)).total_seconds())

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']


def get_activity(Y: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    Y_ = np.argmax(Y, axis=1)
    output = pd.DataFrame(columns=['subject', 'timestamp', 'activity'])
    output['subject'] = t[:, 0]
    output['timestamp'] = t[:, 2]
    output['activity'] = Y_

    return output

def get_intensity(hr: int, info: Dict) -> Tuple[str, float]: # Karvonen formula
    hr_res = (hr - info['HR_min']) / (info['HR_max'] - info['HR_min'])
    hr_res = max(min(hr_res, 1.), 0.)
    act_factor = 1. + hr_res

    if hr_res < 0.2:
        intensity = 'sedentary'
    elif 0.4 > hr_res >= 0.2:
        intensity = 'light'
    elif 0.6 > hr_res >= 0.4:
        intensity = 'moderate'
    elif hr_res >= 0.6:
        intensity = 'vigorous'

    return intensity, act_factor

def get_TEE(intensity_df: pd.DataFrame, info: Dict) -> pd.DataFrame:
    TEE = np.zeros(intensity_df.shape[0]) - 1.
    offset = 0
    for sub_id, sub_df in intensity_df.groupby('subject'):
        sub_info = info[sub_id]
        height, age, weight, sex = sub_info['height'], sub_info['age'], sub_info['weight'], sub_info['sex']
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age + 166 * sex - 161
        TEE[offset: offset + sub_df.shape[0]] = bmr * sub_df['act_factor'].values
        offset += sub_df.shape[0]

    TEE = pd.DataFrame(TEE, columns=['TEE'])
    df = pd.concat([intensity_df, TEE], axis=1)
    return df


class AI_estimator:
    def __init__(self):
        print(get_available_devices())

        self.conf = Parser()
        self.conf.get_args()

        self.builder = builder()

        self.model_name = 'binary_fusion_MIL'
        self.model_dir = os.path.join('models', '%s.h5' % self.model_name)

        self.model = get_fusion_MIL(self.builder.input_shape)

        self.model.compile()
        self.model.summary()
        self.model.load_weights(self.model_dir)

        self.one_batch = True

    def get_intensities(self, activity_df: pd.DataFrame, HR_df: pd.DataFrame, info: Dict) -> pd.DataFrame:
        act_offset = 0
        HR_offset = 0
        intensities = ['undefined' for _ in range(activity_df.shape[0])]
        act_factors = [np.nan for _ in range(activity_df.shape[0])]
        hrs = [0 for _ in range(activity_df.shape[0])]

        for sub_id, sub_act_df in activity_df.groupby('subject'):
            sub_HR_df = HR_df[HR_df['subject'] == sub_id]

            act_timestamps = sub_act_df['timestamp'].values
            HR_timestamps = sub_HR_df['timestamp'].values
            acts = sub_act_df['activity'].values

            for i, (timestamp, act) in enumerate(zip(act_timestamps, acts)):
                dt = get_dist(timestamp, HR_timestamps)
                diffs = np.abs(dt)
                j = np.argmin(diffs)

                if diffs[j] < self.conf.HR_sync_thres:
                    hr = sub_HR_df.loc[j + HR_offset, 'HR']
                    intensity, act_factor = get_intensity(hr, info[sub_id])
                    act_factors[i + act_offset] = act_factor
                    intensities[i + act_offset] = intensity
                    hrs[i + act_offset] = hr

            act_offset += sub_act_df.shape[0]
            HR_offset += sub_HR_df.shape[0]

        intensity_df = pd.DataFrame(intensities, columns=['intensity'])
        act_factors_df = pd.DataFrame(act_factors, columns=['act_factor'])
        hrs_df = pd.DataFrame(hrs, columns=['hr'])
        df = pd.concat([activity_df, intensity_df, act_factors_df, hrs_df], axis=1)

        return df

    def __call__(self, acc, loc, HR, info, verbose: bool = False):

        X, t = self.builder((acc, loc), verbose)
        Y = self.model.predict(X, verbose=0)
        output = get_activity(Y, t)
        output = self.get_intensities(output, HR, info)
        pd.set_option('display.max_columns', None)
        print(output)
        output = get_TEE(output, info)

        return output








