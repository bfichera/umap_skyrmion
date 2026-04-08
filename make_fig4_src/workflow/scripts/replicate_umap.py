import logging
import sys
import time
import argparse
from pathlib import Path

import torch
from numba import config
from workflowrecorder import Recorder
from UMAP_RGB.utils.window import WindowMesh
from UMAP_RGB.networks.EfficientNet_model import EfficientEncoder
from UMAP_RGB.utils.UMAP_RGB import UMAP

from umap_skyrmion import load


parser = argparse.ArgumentParser()
parser.add_argument('--quick-test', action='store_true')
parser.add_argument('--window-length', type=int)
parser.add_argument('--window-stepsize', type=int)
parser.add_argument('--size-fraction', type=float)
parser.add_argument('--frames-fraction', type=float)
parser.add_argument('--trial', type=int)
parser.add_argument('--recorder-path', type=lambda s: Path(s))

_cfg = parser.parse_args()
quick_test = _cfg.quick_test
window_length = _cfg.window_length
window_stepsize = _cfg.window_stepsize
recorder_path = _cfg.recorder_path
frames_fraction = _cfg.frames_fraction
size_fraction = _cfg.size_fraction
trial = _cfg.trial

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
iteration_start_time = time.time()
with Recorder(
        recorder_path.stem,
        recorder_path,
) as recorder:
    start_time = time.time()
    img_stk = load()
    loading_time = time.time()
    recorder.register(loading_time)
    num_frames, size_x, size_y = img_stk.shape
    desired_frames = int(num_frames * frames_fraction)
    desired_x = int(size_x * size_fraction)
    desired_y = int(size_y * size_fraction)
    img_stk = img_stk[:desired_frames, :desired_x, :desired_y]
    splicing_time = time.time()
    recorder.register(img_stk.shape, 'img_stk_shape')
    if quick_test:
        logger.warning('Using wrong size data!')
        mapper_in = img_stk[:20, ::32, ::32]
    else:
        mapper_in = img_stk[:, :, :]
    recorder.register(mapper_in.shape, 'mapper_in_shape')
    recorder.register(splicing_time)
    window_shape = (mapper_in.shape[0], window_length, window_length)
    recorder.register(window_shape)
    step_shape = (
        mapper_in.shape[0], window_stepsize,
        window_stepsize
    )
    recorder.register(step_shape)
    pre_create_windows_time = time.time()
    windows = WindowMesh(mapper_in, window_shape, step_shape)
    post_create_windows_time = time.time()
    recorder.register(pre_create_windows_time)
    recorder.register(post_create_windows_time)

    pre_create_model_time = time.time()
    model = EfficientEncoder(windows, mapper_in)
    post_create_model_time = time.time()
    recorder.register(pre_create_model_time)
    recorder.register(post_create_model_time)
    pre_process_windows_time = time.time()
    windows._window_processor()
    post_process_windows_time = time.time()
    recorder.register(pre_process_windows_time)
    recorder.register(post_process_windows_time)
    pre_extract_embedding_time = time.time()
    low_res_feature_map, upscaler = model.extract_embedding(
        full_output=False
    )
    post_extract_embedding_time = time.time()
    recorder.register(pre_extract_embedding_time)
    recorder.register(post_extract_embedding_time)

    pre_window_ttcf_time = time.time()
    window_ttcf = windows.window_ttcf
    post_window_ttcf_time = time.time()
    recorder.register(pre_window_ttcf_time)
    recorder.register(post_window_ttcf_time)
    recorder.register(window_ttcf.shape, 'window_ttcf_shape')

    mapper = UMAP(low_res_feature_map, upscaler)
    pre_rgb_time = time.time()
    mapper.generate_rgb(sparsity_mult=20)
    post_rgb_time = time.time()
    recorder.register(pre_rgb_time)
    recorder.register(post_rgb_time)

    # recorder.register(
    #     mapper.rgb[0],
    #     name='mapper_get_rgb_0',
    #     description='mapper.rgb[0]',
    # )
    # recorder.register(
    #     mapper.rgb,
    #     name='mapper_rgb',
    #     description='mapper.rgb',
    # )
    # recorder.register(
    #     mapper.low_res_rgb,
    #     name='mapper_low_res_rgb',
    #     description='mapper.low_res_rgb',
    # )
    end_time = time.time()
    recorder.register(start_time)
    recorder.register(end_time)
