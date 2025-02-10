from os import eventfd_write

import pandas as pd
import numpy as np
from geopy.distance import great_circle
from typing import *


def add_norm_xyz(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['norm_xyz'] = np.sqrt(x['acc_x'] ** 2 + x['acc_y'] ** 2 + x['acc_z'] ** 2)

    return x


def add_norm_xy(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['norm_xy'] = np.sqrt(x['acc_x'] ** 2 + x['acc_y'] ** 2)

    return x


def add_norm_yz(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['norm_yz'] = np.sqrt(x['acc_y'] ** 2 + x['acc_z'] ** 2)

    return x


def add_norm_xz(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['norm_xz'] = np.sqrt(x['acc_x'] ** 2 + x['acc_z'] ** 2)

    return x


def add_jerk(x: pd.DataFrame, fillna: bool = True) -> pd.DataFrame:
    x = x.copy()

    groups = x.groupby(['subject', 'session'])
    acc_dx = (groups['acc_x'].diff() / groups['timestamp'].diff().dt.total_seconds()).values[:, np.newaxis]
    acc_dy = (groups['acc_y'].diff() / groups['timestamp'].diff().dt.total_seconds()).values[:, np.newaxis]
    acc_dz = (groups['acc_z'].diff() / groups['timestamp'].diff().dt.total_seconds()).values[:, np.newaxis]

    acc_di = np.concatenate((acc_dx, acc_dy, acc_dz), axis=1)
    jerk = np.sqrt(np.sum(np.square(acc_di), axis=1))

    x['jerk'] = jerk
    groups = x.groupby(['subject', 'session'])

    if fillna:
        mask = groups.cumcount() == 0
        x['jerk'] = x['jerk'].where(~mask, 0)

    return x

def add_azimuth(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x['az_angle'] = np.arctan2(x['acc_y'], x['acc_x'])

    return x

def add_elevation(x: pd.DataFrame) -> np.float32:
    x = x.copy()

    x['el_angle'] = np.arctan2(x['acc_z'], np.sqrt(x['acc_x'] ** 2 + x['acc_y'] ** 2))

    return x

def calc_velocity(lat: np.ndarray, long: np.ndarray, time: np.ndarray, i) -> np.float32:
    if np.isnan([lat[i], long[i], lat[i - 1], long[i - 1]]).any():
        return np.nan

    point1 = (lat[i-1], long[i-1])
    point2 = (lat[i], long[i])
    ds = great_circle(point1, point2).m
    dt = (time[i] - time[i - 1]) / np.timedelta64(1, 's')

    velocity = ds / dt
    return velocity

def get_velocity(x: pd.DataFrame) -> np.ndarray:
    lat = x['latitude'].values
    long = x['longitude'].values
    time = x['timestamp'].values

    velocity = np.array([calc_velocity(lat, long, time, i) for i in range(1, len(x))])
    velocity = np.concatenate(([np.nan], velocity))

    return velocity

def add_velocity(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    groups = x.groupby('subject')
    x['velocity'] = groups.apply(lambda g: get_velocity(g)).values.squeeze()

    return x


def calc_acceleration(lat, long, time, i) -> pd.DataFrame:
    vel1 = calc_velocity(lat, long, time, i-1)
    vel2 = calc_velocity(lat, long, time, i)
    if np.isnan([vel1, vel2]).any():
        return np.nan

    dv = vel2 - vel1
    dt = (time[i] - time[i - 1]) / np.timedelta64(1, 's')
    acceleration = dv / dt

    return acceleration

def get_acceleration(x: pd.DataFrame) -> np.ndarray:
    lat = x['latitude'].values
    long = x['longitude'].values
    time = x['timestamp'].values

    acceleration = np.array([calc_acceleration(lat, long, time, i) for i in range(2, len(x))])
    acceleration = np.concatenate(([np.nan, np.nan], acceleration))

    return acceleration

def add_acceleration(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    groups = x.groupby('subject')
    x['acceleration'] = groups.apply(lambda g: get_acceleration(g)).values.squeeze()

    return x


def calc_bearing(lat: np.ndarray, long: np.ndarray, i) -> np.float32:
    if np.isnan([lat[i], long[i], lat[i - 1], long[i - 1]]).any():
        return np.nan

    y = np.sin(long[i] - long[i - 1]) * np.cos(lat[i])
    x = np.cos(lat[i - 1]) * np.sin(lat[i]) - \
        np.sin(lat[i - 1]) * np.cos(long[i] - long[i - 1]) * np.cos(lat[i])

    angle = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return angle


def calc_bearing_rate(lat: np.ndarray, long: np.ndarray, time: np.ndarray, i) -> np.float32:
    angle1 = calc_bearing(lat, long, i - 1)
    angle2 = calc_bearing(lat, long, i)
    return 1000. * abs(angle2 - angle1) / (time[i] - time[i - 1])


def get_bearing(x: pd.DataFrame) -> np.ndarray:
    lat = x['latitude'].values
    long = x['longitude'].values
    time = x['timestamp'].values

    bearing_rate = np.array([calc_bearing(lat, long, time, i) for i in range(2, len(x))])
    bearing_rate = np.concatenate(([np.nan, np.nan], bearing_rate))

    return bearing_rate


def add_bearing(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    groups = x.groupby(['subject', 'session'])
    bearing = groups.apply(lambda g: pd.Series(get_bearing(g), index=g.index))
    x['bearing'] = bearing.reset_index(level=['subject', 'session'], drop=True)

    return x






