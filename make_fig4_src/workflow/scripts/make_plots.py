from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def stderr(a):
    return np.std(a) / len(a)


# vs. number of frames
# vs. number of windows
# computing the ttcfs
# doing the efficientnet+umap part

parser = argparse.ArgumentParser()
parser.add_argument('--csv-path', type=lambda s: Path(s))
parser.add_argument('--output', type=lambda s: Path(s))
_cfg = parser.parse_args()
output_dir = _cfg.output
csv_path = _cfg.csv_path

with open(csv_path, 'r') as fh:
    d = pd.read_csv(fh)

label_cols = ['size_fraction', 'frames_fraction', 'window_size', 'window_step_size', 'num_frames', 'size_x', 'size_y', 'num_windows_x', 'num_windows_y']
measure_cols = [c for c in d.columns if c.endswith('_time')]


# Times vs num_frames for a certain window size and spatial size
a = d.loc[
    (d['window_size'] == 65)
    & (d['window_step_size'] == 10)
    & (d['size_fraction'] == 1.0)
][['num_frames']+measure_cols].groupby('num_frames').agg(['mean', 'std', stderr, 'size'])
for col in measure_cols:
    fname = 'num_frames_vs_' + col + '.csv'
    path = output_dir / fname
    g = pd.DataFrame(
        {
            'x': a.index.to_numpy(),
            'y': a[col, 'mean'].to_numpy(),
            'yerr': a[col, 'stderr'].to_numpy()
        },
    )
    g.to_csv(path, index=False)

# Times vs window_size for a certain number of frames and spatial size
a = d.loc[
    (d['frames_fraction'] == 1.0)
    & (d['window_step_size'] == 10)
    & (d['size_fraction'] == 1.0)
][['num_windows_x']+measure_cols].groupby('num_windows_x').agg(['mean', 'std', stderr, 'size'])
for col in measure_cols:
    fname = 'num_windows_x_vs_' + col + '.csv'
    path = output_dir / fname
    g = pd.DataFrame(
        {
            'x': a.index.to_numpy(),
            'y': a[col, 'mean'].to_numpy(),
            'yerr': a[col, 'stderr'].to_numpy()
        },
    )
    g.to_csv(path, index=False)

# Times vs spatial size for a certain number of frames and window size
a = d.loc[
    (d['frames_fraction'] == 1.0)
    & (d['window_step_size'] == 10)
    & (d['window_size'] == 65)
][['size_x']+measure_cols].groupby('size_x').agg(['mean', 'std', stderr, 'size'])
for col in measure_cols:
    fname = 'size_x_vs_' + col + '.csv'
    path = output_dir / fname
    g = pd.DataFrame(
        {
            'x': a.index.to_numpy(),
            'y': a[col, 'mean'].to_numpy(),
            'yerr': a[col, 'stderr'].to_numpy()
        },
    )
    g.to_csv(path, index=False)
