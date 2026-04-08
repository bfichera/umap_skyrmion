from pathlib import Path
import argparse

import pandas as pd


# vs. number of frames
# vs. number of windows
# computing the ttcfs
# doing the efficientnet+umap part

parser = argparse.ArgumentParser()
parser.add_argument('--csv-path', type=lambda s: Path(s))
parser.add_argument('--output-path', type=lambda s: Path(s))
_cfg = parser.parse_args()
output_path = _cfg.output_path
csv_path = _cfg.csv_path

with open('csv_path', 'r') as fh:
    d = pd.read_csv(fh)
