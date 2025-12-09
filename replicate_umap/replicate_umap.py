import logging
import sys
import time
import argparse
from pathlib import Path
import itertools

import torch
from numba import config
from workflowrecorder import Recorder
from UMAP_RGB.utils.window import WindowMesh
from UMAP_RGB.networks.EfficientNet_model import EfficientEncoder
from UMAP_RGB.utils.UMAP_RGB import UMAP

from umap_skyrmion import load


parser = argparse.ArgumentParser()
parser.add_argument('--quick-test', action='store_true')
parser.add_argument('--half-window-lengths', type=int, nargs='+')
parser.add_argument('--window-stepsize-ratios', type=int, nargs='+')

_cfg = parser.parse_args()
quick_test = _cfg.quick_test
window_lengths = [c * 2 for c in _cfg.half_window_lengths]
window_stepsize_ratios = _cfg.window_stepsize_ratios

start_time = time.time()

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

logger.info(f"Numba thread layer: {config.THREADING_LAYER}")
logger.info(f"Numba threads: {config.NUMBA_NUM_THREADS}")
logger.info(f"PyTorch intra-op threads: {torch.get_num_threads()}")
logger.info(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")

results_path = Path.cwd() / 'results'
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)
for window_length, window_stepsize_ratio in itertools.product(
        window_lengths, window_stepsize_ratios):
    iteration_start_time = time.time()
    recorder_path = (
        results_path / f'{window_length:03d}'
        f'_{window_stepsize_ratio:03d}.pkl'
    )
    if recorder_path.exists():
        logger.warning(f'Skipping {recorder_path} because it already exists.')
        continue
    with Recorder(
            recorder_path.stem,
            recorder_path,
    ) as recorder:
        img_stk = load()
        recorder.register(img_stk)
        if quick_test:
            logger.warning('Using wrong size data!')
            mapper_in = img_stk[:20, ::32, ::32]
        else:
            mapper_in = img_stk[:, :, :]
        recorder.register(mapper_in)
        window_shape = (mapper_in.shape[0], window_length, window_length)
        recorder.register(window_shape)
        step_shape = (
            mapper_in.shape[0], window_length // window_stepsize_ratio,
            window_length // window_stepsize_ratio
        )
        recorder.register(step_shape)
        windows = WindowMesh(mapper_in, window_shape, step_shape)

        model = EfficientEncoder(windows, mapper_in)
        windows._window_processor()
        low_res_feature_map, upscaler = model.extract_embedding(
            full_output=False
        )

        recorder.register(windows.window_ttcf, name='window_ttcf')

        mapper = UMAP(low_res_feature_map, upscaler)
        mapper.generate_rgb(sparsity_mult=20)

        recorder.register(
            mapper.rgb[0],
            name='mapper_get_rgb_0',
            description='mapper.rgb[0]',
        )
        recorder.register(
            mapper.rgb,
            name='mapper_rgb',
            description='mapper.rgb',
        )
        recorder.register(
            mapper.low_res_rgb,
            name='mapper_low_res_rgb',
            description='mapper.low_res_rgb',
        )

    logger.info(
        'Single window length / stepsize combination finished after '
        f'{time.time() - iteration_start_time} seconds.'
    )

logger.info(
    f'Entire script finished after {time.time() - start_time} seconds.'
)
