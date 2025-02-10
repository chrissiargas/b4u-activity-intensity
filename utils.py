import math
from typing import List

import numpy as np
import pandas as pd
from features import *

to_seconds = np.vectorize(lambda x: x.total_seconds())

def assign_session(x: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    id = 1
    start = 0
    for gap in gaps:
        end = x[x['gap_id'] == gap].index[0]
        x.loc[start:end, 'session'] = id
        start = x[x['gap_id'] == gap].index[-1] + 1
        id += 1
    x.loc[start:, 'session'] = id

    return x

def acc_form(x: pd.DataFrame, threshold: float) -> pd.DataFrame:
    x = x.copy()

    # Identify missing data groups
    x["missing"] = x.isna().any(axis=1).values.astype(int).astype(bool)
    x["session"] = 0

    # Compute the duration of each gap
    x["gap_shift"] = x["missing"].shift(1, fill_value=False)  # Identify start of gaps
    x["gap_id"] = (x["gap_shift"].ne(x["missing"]).cumsum() + 1) * x["missing"]  # Assign gap IDs
    gap_durations = x.groupby("gap_id")["timestamp"].agg(lambda x: (x.max() - x.min()).total_seconds())

    # Identify big gaps > threshold (s)
    big_gaps = gap_durations[gap_durations > threshold].index[1:] # exclude 0, corresponding to valid data
    x = assign_session(x, big_gaps)

    # Drop big gaps
    x = x[~x["gap_id"].isin(big_gaps)].drop(columns=["missing", "gap_shift", "gap_id"])

    # Interpolate small gaps within each session
    x[["acc_x", "acc_y", "acc_z"]] = x.groupby("session")[["acc_x", "acc_y", "acc_z"]].transform(lambda g: g.interpolate())
    x = x.reset_index()

    return x

def acc_produce(x: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    x = x.copy()

    if features is None:
        return x

    if 'norm_xyz' in features:
        x = add_norm_xyz(x)

    if 'norm_xy' in features:
        x = add_norm_xy(x)

    if 'norm_xz' in features:
        x = add_norm_xz(x)

    if 'norm_yz' in features:
        x = add_norm_yz(x)

    if 'jerk' in features:
        x = add_jerk(x)

    if 'azimuth' in features:
        x = add_azimuth(x)

    if 'elevation' in features:
        x = add_elevation(x)

    return x

def loc_produce(x: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    x = x.copy()

    if features is None:
        return x

    if 'velocity' in features:
        x = add_velocity(x)

    if 'acceleration' in features:
        x = add_acceleration(x)

    if 'bearing' in features:
        x = add_bearing(x)

    return x

def segment(x: pd.DataFrame, length: int, step: int, start: int = 0) -> np.ndarray:
    X = x.values[start:]

    n_segments = math.ceil((X.shape[0] - length) / step)
    n_segments = max(0, n_segments)

    X = np.lib.stride_tricks.as_strided(X,
                                        shape=(n_segments, length, X.shape[1]),
                                        strides=(step * X.strides[0], X.strides[0], X.strides[1]))

    return X

def acc_segment(x: pd.DataFrame, length: int, stride: int, bag_size: int, bag_stride: int, start: int = 0) -> \
    Tuple[np.ndarray, np.ndarray, Dict]:

    x = x.copy()

    full_length = length + (bag_size - 1) * bag_stride
    features = x.columns[x.columns.str.contains('acc|norm|jerk|angle')]
    channels = {k: v for v, k in enumerate(features)}

    groups = x.groupby(['subject', 'session'])
    x_segs = groups.apply(lambda g: segment(g[features], full_length, stride, start))
    x_segs = np.concatenate(x_segs.values)

    t_features = ['subject', 'session', 'timestamp']
    t_segs = groups.apply(lambda g: segment(g[t_features], full_length, stride, start))
    t_segs = np.concatenate(t_segs.values)
    t_info = np.concatenate((t_segs[:, 0, :-1], t_segs[:, full_length // 2, [-1]]), axis=1)

    return x_segs, t_info, channels


def loc_segment(x: pd.DataFrame, length: int, stride: int, start: int = 0) -> \
        Tuple[np.ndarray, np.ndarray, Dict]:
    x = x.copy()

    features = x.columns[x.columns.str.contains('latitude|longitude|velocity|acceleration|bearing')]
    channels = {k: v for v, k in enumerate(features)}

    groups = x.groupby('subject')
    x_segs = groups.apply(lambda g: segment(g[features], length, stride, start))
    x_segs = np.concatenate(x_segs.values)

    t_features = ['subject', 'timestamp']
    t_segs = groups.apply(lambda g: segment(g[t_features], length, stride, start))
    t_segs = np.concatenate(t_segs.values)
    t_info = np.concatenate((t_segs[:, 0, :-1], t_segs[:, length // 2, [-1]]), axis=1)

    return x_segs, t_info, channels

def bagging(acc_t: np.ndarray, loc_t: np.ndarray, threshold: float) -> List[int]:
    syncing = [-1 for _ in range(len(acc_t))]

    subs = np.unique(acc_t[:,0])
    for sub in subs:
        sub_acc_t = acc_t[acc_t[:, 0] == sub]
        sub_loc_t = loc_t[loc_t[:, 0] == sub]
        loc_timestamps = sub_loc_t[:, -1]
        acc_timestamps = sub_acc_t[:, -1]

        for i, timestamp in enumerate(acc_timestamps):
            dt = to_seconds(timestamp - loc_timestamps)

            diffs = np.abs(dt)
            j = np.argmin(diffs)

            if diffs[j] < threshold:
                syncing[i] = j

    return syncing








