import pickle
import argparse
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt


def yn_input(prompt, default):
    default_str = '[y/N]'
    if default is True:
        default_str = '[Y/n]'
    while True:
        x = input(f'{prompt} {default_str}')
        if x.lower().startswith('y'):
            return True
        if x.lower().startswith('n'):
            return False
        if x == '':
            return default
        print('Please try again.')


def _normalize(a):
    return (a - np.amin(a)) / (np.amax(a) - np.amin(a))


parser = argparse.ArgumentParser()
parser.add_argument('record_path', type=lambda s: Path(s))
parser.add_argument('output_path', type=lambda s: Path(s))
parser.add_argument('--num_frames', type=int, default=5)
parser.add_argument('--ttcfs_i', nargs='+', type=int)
parser.add_argument('--ttcfs_j', nargs='+', type=int)
parser.add_argument('--debug', action='store_true')

_cfg = parser.parse_args()
record_path = _cfg.record_path
output_path = _cfg.output_path
num_frames = _cfg.num_frames
debug = _cfg.debug
ttcf_idxs = np.array(list(zip(_cfg.ttcfs_i, _cfg.ttcfs_j)))

if debug:
    breakpoint()

if output_path.exists():
    if not yn_input(
            f'This will delete {output_path.absolute().as_posix()}. Continue?',
            default=False):
        exit()
    shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

with open(record_path, 'rb') as fh:
    record = pickle.load(fh)
    img_stk = record.img_stk[:, :, :]
    frames_step = img_stk.shape[0] // num_frames
    frames = _normalize(img_stk[::frames_step, :, :])
    for i in range(frames.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(frames[i],cmap='magma')
        ax.axis('off')
        plt.savefig(
            output_path / f'frame_{i}.pdf',
            format='pdf',
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

    window_ttcf = record.window_ttcf
    for ttcf_i, ttcf_j in ttcf_idxs:
        ttcf = window_ttcf[0, ttcf_i, ttcf_j, :, :]
        fig, ax = plt.subplots()
        ax.imshow(ttcf,origin='lower')
        ax.axis('off')
        plt.savefig(
            output_path / f'ttcf_{ttcf_i}_{ttcf_j}.pdf',
            format='pdf',
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

    mapper_get_rgb_0 = record.mapper_get_rgb_0
    fig, ax = plt.subplots()
    ax.imshow(mapper_get_rgb_0)
    ax.axis('off')
    plt.savefig(
        output_path / 'mapper_get_rgb_0.pdf',
        format='pdf',
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)

    # img_stk = load()
    # recorder.register(img_stk)
    # if quick_test:
    #     logger.warning('Using wrong size data!')
    #     mapper_in = img_stk[:20, ::32, ::32]
    # else:
    #     mapper_in = img_stk[:, :, :]
    # recorder.register(mapper_in)
    # window_shape = (mapper_in.shape[0], window_length, window_length)
    # recorder.register(window_shape)
    # step_shape = (
    #     mapper_in.shape[0], window_length // window_stepsize_ratio,
    #     window_length // window_stepsize_ratio
    # )
    # recorder.register(step_shape)
    # windows = WindowMesh(mapper_in, window_shape, step_shape)

    # model = EfficientEncoder(windows, mapper_in)
    # windows._window_processor()
    # low_res_feature_map, upscaler = model.extract_embedding(
    #     full_output=False
    # )

    # This is where we would create the schematic UMAP images...if we knew
    # how to do that correctly.
    # recorder.register(windows.window_ttcf, name='window_ttcf')

    # mapper = UMAP(low_res_feature_map, upscaler)
    # mapper.generate_rgb(sparsity_mult=20)

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
