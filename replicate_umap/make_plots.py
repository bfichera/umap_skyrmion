from pathlib import Path
import re
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
parser.add_argument('--normalizer', type=str, choices=['michelson', 'rms'])
_cfg = parser.parse_args()
do_show = _cfg.show
normalizer = _cfg.normalizer
if do_show:
    show = plt.show
else:

    def show():
        pass


for path in tqdm(list((Path.cwd() / f'results_{normalizer}').glob('test2_results_*_*.pkl'))):
    window_length, window_stepsize_ratio = (
        int(s) for s in
        re.match(r'^test2_results_([0-9]*)_([0-9]*).pkl$', path.name).groups()
    )
    plots_folder = Path.cwd() / f'plots_{normalizer}' / str(window_length
                                              ) / str(window_stepsize_ratio)
    plots_folder.mkdir(parents=True, exist_ok=True)
    with open(path, 'rb') as fh:
        r = pickle.load(fh)
    plt.imshow(r.mapper_low_res_rgb[0, :, :, :])
    plt.savefig(plots_folder / 'low_res_rgb')
    show()
    plt.imshow(r.mapper_rgb[0, :, :, :])
    plt.savefig(plots_folder / 'rgb')
    show()
    plt.imshow(r.img_stk[0, 0, :, :])
    plt.savefig(plots_folder / 'img_stk0')
    show()
    plt.imshow(r.img_stk[0, 1, :, :])
    plt.savefig(plots_folder / 'img_stk1')
    show()
    plt.imshow(np.sum(r.img_stk[:, 0, :, :], axis=0))
    plt.savefig(plots_folder / 'sum_img_stk0')
    show()
    plt.imshow(np.sum(r.img_stk[:, 1, :, :], axis=0))
    plt.savefig(plots_folder / 'sum_img_stk1')
    show()
    plt.imshow(
        np.absolute(np.fft.fftshift(np.fft.fft2(r.img_stk[0, 0, :, :])))
    )
    plt.savefig(plots_folder / 'fft_img_stk0')
    show()
    plt.imshow(
        np.absolute(np.fft.fftshift(np.fft.fft2(r.img_stk[0, 1, :, :])))
    )
    plt.savefig(plots_folder / 'fft_img_stk1')
    show()

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    u0, u1, u2, u3 = r.mapper_low_res_rgb.shape
    collapsed_rgb = r.mapper_low_res_rgb.reshape(u1 * u2, 3)
    ax.scatter(*(collapsed_rgb[:, i] for i in range(3)))
    ax.set_xlabel('r')
    ax.set_ylabel('g')
    ax.set_zlabel('b')
    plt.savefig(plots_folder / 'rgb_scatter')
    show()
    plt.close()

    std = np.std(r.window_ttcf, axis=(-1, -2))[0, :, :]
    mean = np.mean(r.window_ttcf, axis=(-1, -2))[0, :, :]
    plt.imshow((std / mean).round(5))
    plt.savefig(plots_folder / 'rms_contrast')
    show()
    plt.close()

    s0, s1, s2, s3, s4 = r.window_ttcf.shape
    t = r.window_ttcf.reshape(s0, s1, s2, s3 * s4)
    plt.imshow(
        (
            (t.max(axis=-1) - t.min(axis=-1)) /
            (t.max(axis=-1) + t.min(axis=-1))
        )[0, :, :]
    )
    plt.savefig(plots_folder / 'michelson_contrast')
    show()
    plt.close()
