import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import *

figpath = './figs'
to_seconds = np.vectorize(lambda x: x.total_seconds())

def plot_window(x: np.ndarray, t: np.ndarray, channels: Dict, features: List,
                subject: Optional[str] = None, session: Optional[int] = None, timestamp: str = ''):
    indices = [channels[feature] for feature in features]
    x = x[..., indices]

    timestamp = np.array([pd.to_datetime(timestamp)],dtype=pd.Timestamp)

    if subject is not None:
        idx = np.argwhere(t[:, 0] == subject).squeeze()
        x = x[idx]
        t = t[idx]

    if session is not None:
        idx = np.argwhere(t[:, 1] == session).squeeze()
        x = x[idx]
        t = t[idx]

    idx = np.argwhere(np.abs(to_seconds(t[:, -1] - timestamp)) < 60).squeeze()

    if len(idx.shape) > 0:
        idx = idx[-1]

    window = x[idx]

    fig, axs = plt.subplots(1, sharex=True, figsize=(20, 15))
    axs.plot(window, linewidth=0.5, label = features)

    plt.legend()
    filepath = os.path.join(figpath, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
    plt.savefig(filepath, format="png", bbox_inches="tight")
    plt.close()
