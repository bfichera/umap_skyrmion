from pathlib import Path
import pickle
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

data_cmap = 'Greys'
cmap = 'viridis'
ttcf_cmap = 'magma'

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=lambda s: Path(s))
parser.add_argument('--show', action='store_true')
parser.add_argument('--output-dir', type=lambda s: Path(s))
parser.add_argument('--extension', type=str, default='.pdf')
parser.add_argument('--ttcfs_i', nargs='+', type=float)
parser.add_argument('--ttcfs_j', nargs='+', type=float)
_cfg = parser.parse_args()
do_show = _cfg.show
output_dir = _cfg.output_dir
results_path = _cfg.results_path
extension = _cfg.extension
ttcf_idxs = np.array(list(zip(_cfg.ttcfs_i, _cfg.ttcfs_j)))

kwargs = {
    'bbox_inches': 'tight',
    'pad_inches': 0,
}

params = {}

if do_show:
    show = plt.show
else:

    def show():
        pass

plots_folder = _cfg.output_dir
with open(results_path, 'rb') as fh:
    r = pickle.load(fh)
plt.imshow(r.mapper_low_res_rgb[0, :, :, :], cmap=data_cmap, origin='lower')
plt.axis('off')
plt.savefig(plots_folder / f'low_res_rgb{extension}', **kwargs)
show()
plt.imshow(r.mapper_rgb[0, :, :, :], cmap=data_cmap, origin='lower')
plt.axis('off')
plt.savefig(plots_folder / f'rgb{extension}', **kwargs)
show()

num_frames = r.img_stk.shape[0]
frames_of_interest = np.linspace(0, num_frames-1, 10, endpoint=True, dtype=int)
cdw_vmin_0 = np.inf
cdw_vmax_0 = -np.inf
cdw_vmin_1 = np.inf
cdw_vmax_1 = -np.inf
for frame in frames_of_interest:
    im = r.img_stk[frame, :, :]
    if np.amin(im) < cdw_vmin_0:
        cdw_vmin_0 = np.amin(im)
    if np.amax(im) > cdw_vmax_0:
        cdw_vmax_0 = np.amax(im)
params['img_stk_0_vmin'] = cdw_vmin_0
params['img_stk_0_vmax'] = cdw_vmax_0
for frame in frames_of_interest:
    mappable_0 = plt.imshow(r.img_stk[frame, :, :], cmap=data_cmap, vmin=cdw_vmin_0, vmax=cdw_vmax_0, origin='lower')
    plt.axis('off')
    plt.savefig(plots_folder / f'img_stk_{frame:03d}_0{extension}', **kwargs)
    show()

fig_cbar, ax_cbar = plt.subplots(figsize=(1.5, 6))
cbar = fig_cbar.colorbar(mappable_0, cax=ax_cbar)
cbar.set_ticks([])
plt.savefig(plots_folder / f'img_stk_0_colorbar{extension}', **kwargs)
plt.close()

ttcf_vmin = np.inf
ttcf_vmax = -np.inf
for ttcf_i, ttcf_j in ttcf_idxs:
    n_i = r.window_ttcf.shape[1]
    n_j = r.window_ttcf.shape[2]
    ttcf_i_idx = int(ttcf_i * n_i)
    ttcf_j_idx = int(ttcf_j * n_j)
    ttcf = r.window_ttcf[0, ttcf_i_idx, ttcf_j_idx, :, :]
    if np.amax(ttcf) > ttcf_vmax:
        ttcf_vmax = np.amax(ttcf)
    if np.amin(ttcf) < ttcf_vmin:
        ttcf_vmin = np.amin(ttcf)
ttcf_vmin = 0
params['ttcf_vmin'] = ttcf_vmin
params['ttcf_vmax'] = ttcf_vmax
for ttcf_i, ttcf_j in ttcf_idxs:
    n_i = r.window_ttcf.shape[1]
    n_j = r.window_ttcf.shape[2]
    ttcf_i_idx = int(ttcf_i * n_i)
    ttcf_j_idx = int(ttcf_j * n_j)
    ttcf = r.window_ttcf[0, ttcf_i_idx, ttcf_j_idx, :, :]
    mappable = plt.imshow(ttcf, cmap=ttcf_cmap, vmin=ttcf_vmin, vmax=ttcf_vmax, origin='lower')
    plt.axis('off')
    plt.savefig(plots_folder / f'ttcf_{ttcf_i_idx}_{ttcf_j_idx}{extension}', **kwargs)
fig_cbar, ax_cbar = plt.subplots(figsize=(1.5, 6))
cbar = fig_cbar.colorbar(mappable, cax=ax_cbar)
cbar.set_ticks([])
plt.savefig(plots_folder / 'ttcf_colorbar.pdf', **kwargs)
plt.close()

im = np.sum(r.img_stk[:, :, :], axis=0)
mappable = plt.imshow(im, cmap=cmap, vmin=np.amin(im), vmax=np.amax(im), origin='lower')
params['sum_img_stk_0_vmin'] = np.amin(im)
params['sum_img_stk_0_vmax'] = np.amax(im)
plt.axis('off')
plt.savefig(plots_folder / f'sum_img_stk0{extension}', **kwargs)
show()
fig_cbar, ax_cbar = plt.subplots(figsize=(1.5, 6))
cbar = fig_cbar.colorbar(mappable, cax=ax_cbar)
cbar.set_ticks([])
plt.savefig(plots_folder / 'sum_img_stk_0_colorbar.pdf', **kwargs)
plt.close()
fft_im = np.absolute(np.fft.fftshift(np.fft.fft2(r.img_stk[0, :, :])))
plt.imshow(
    fft_im,
    vmin=0.0,
    vmax=np.nanpercentile(fft_im, 99),
    cmap=cmap,
    origin='lower',
)
plt.savefig(plots_folder / f'fft_img_stk0{extension}', **kwargs)
show()

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
u0, u1, u2, u3 = r.mapper_low_res_rgb.shape
collapsed_rgb = r.mapper_low_res_rgb.reshape(u1 * u2, 3)
ax.scatter(*(collapsed_rgb[:, i] for i in range(3)))
ax.set_xlabel('r')
ax.set_ylabel('g')
ax.set_zlabel('b')
plt.savefig(plots_folder / f'rgb_scatter{extension}', **kwargs)
show()
plt.close()

s0, s1, s2, s3, s4 = r.window_ttcf.shape
plt.imshow(
    np.std(r.window_ttcf.reshape(s0, s1, s2, s3 * s4), axis=-1)[0, :, :],
    cmap=cmap, 
    origin='lower',
)
plt.savefig(plots_folder / f'rms_contrast{extension}', **kwargs)
show()
plt.close()

t = r.window_ttcf.reshape(s0, s1, s2, s3 * s4)
plt.imshow(
    (
        (t.max(axis=-1) - t.min(axis=-1)) /
        (t.max(axis=-1) + t.min(axis=-1))
    )[0, :, :],
    cmap=cmap,
    origin='lower',
)
plt.savefig(plots_folder / f'michelson_contrast{extension}', **kwargs)
show()
plt.close()

with open(plots_folder / 'params.json', 'w') as fh:
    for k, v in params.items():
        if isinstance(v, (np.float32, np.float64)):
            params[k] = v.item()
    json.dump(params, fh)
