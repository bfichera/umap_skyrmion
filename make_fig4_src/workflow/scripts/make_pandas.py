from pathlib import Path
import pickle
import argparse

import pandas as pd


# vs. number of frames
# vs. number of windows
# computing the ttcfs
# doing the efficientnet+umap part

parser = argparse.ArgumentParser()
parser.add_argument('--results-path', type=lambda s: Path(s))
parser.add_argument('--output-path', type=lambda s: Path(s))
_cfg = parser.parse_args()
\utput_path = _cfg.output_path
results_path = _cfg.results_path

rows = []
for trial_d in results_path.glob('trial_*'):
    trial = int(trial_d.name.split('_')[-1])
    for size_fraction_d in trial_d.glob('size_fraction_*'):
        size_fraction = float(size_fraction_d.name.split('_')[-1])
        for frames_fraction_d in size_fraction_d.glob('frames_fraction_*'):
            frames_fraction = float(frames_fraction_d.name.split('_')[-1])
            for window_size_d in frames_fraction_d.glob('window_size_*'):
                window_size = int(window_size_d.name.split('_')[-1])
                for window_step_size_d in window_size_d.glob('window_step_size_*'):
                    window_step_size = int(window_step_size_d.name.split('_')[-1])
                    with open(window_step_size_d / 'results.pkl', 'rb') as fh:
                        r = pickle.load(fh)
                    row = {
                        'trial': trial,
                        'size_fraction': size_fraction,
                        'frames_fraction': frames_fraction,
                        'window_size': window_size,
                        'window_step_size': window_step_size,
                        'num_frames': r.img_stk_shape[0],
                        'size_x': r.img_stk_shape[1],
                        'size_y': r.img_stk_shape[2],
                        'num_windows_x': r.window_ttcf_shape[2],
                        'num_windows_y': r.window_ttcf_shape[3],
                        'total_time': r.end_time - r.start_time,
                        'generate_rgb_time': r.post_rgb_time - r.pre_rgb_time,
                        'window_ttcf_time': r.post_window_ttcf_time - r.pre_window_ttcf_time,
                        'extract_embedding_time': r.post_extract_embedding_time - r.pre_extract_embedding_time,
                        'process_windows_time': r.post_process_windows_time - r.pre_process_windows_time,
                        'create_model_time': r.post_create_model_time - r.pre_create_model_time,
                        'create_windows_time': r.post_create_windows_time - r.pre_create_windows_time,
                    }
                    rows.append(row)
d = pd.DataFrame(rows)
with open(output_path, 'w') as fh:
    d.to_csv(fh, index=False)
