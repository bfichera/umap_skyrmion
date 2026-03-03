#!/usr/bin/env python

# Added by Bryan
# Parse command-line arguments
# Add PCA support
import argparse
from pathlib import Path
import psutil
import os
from sklearn.decomposition import PCA as skPCA

parser = argparse.ArgumentParser()
parser.add_argument('--window-size', type=int, default=16)
parser.add_argument('--window-step-size', type=int, default=16)
parser.add_argument('--feature-vecs-path', type=lambda s: Path(s))
parser.add_argument('--output-file', type=lambda s: Path(s))
parser.add_argument('--mapper', type=str, choices=['umap', 'pca'], default='umap')
_cfg = parser.parse_args()
window_size = _cfg.window_size
window_step_size = _cfg.window_step_size
feature_vecs_path = _cfg.feature_vecs_path
output_file = _cfg.output_file
mapper_str = _cfg.mapper

# Added by Bryan
# Diagnose memory usage
def mem(label=""):
    with open("/proc/self/status") as f:
        lines = f.readlines()
    vals = {}
    for l in lines:
        if l.startswith(("VmRSS", "VmHWM", "VmSize", "VmPeak")):
            k, v = l.split(":")
            vals[k] = v.strip()
    print(f"[PROC] {label}:", vals)


# coding: utf-8

# ## Imports

# In[28]:

# File/data control

# Workhorses

import numpy as np

# Misc.
from tqdm import tqdm
from joblib import delayed, Parallel

# Clustering / Data Analysis

from sklearn.preprocessing import StandardScaler
import umap

# Neural Networks

import torch
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torchvision.transforms.functional as F

# Imaging / Plotting

import tifffile
from PIL import Image  # Pillow library for image manipulation
import matplotlib as mpl
import matplotlib.pyplot as plt

import os

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# Bryan commented this out
# os.environ["OMP_NUM_THREADS"] = "64"

mpl.rcParams['figure.dpi'] = 200

# import all plotting packages

plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.size'] = 8
plt.rcParams['font.sans-serif'] = ['Arial']

# For papers -
# All figs will be 14 inch for two columns and 7 in for one column
plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['image.interpolation'] = 'none'

plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.labelleft'] = True

# ## Utility

# In[53]:


def autocorr(waterfall, n_proc=8, verbose=0):
    I_bar_t = np.mean(waterfall, axis=1)
    I_bar_t1_I_bar_t2 = np.outer(I_bar_t, I_bar_t)
    # TODO
    # Note, change to specify CORRECT AXES OVER WHICH TO MEAN - should be
    # vector which outer product gives array, no?
    variance_squared = np.mean(np.square(waterfall)) - np.mean(waterfall)**2

    n_frames = waterfall.shape[0]

    ab = np.zeros((n_frames, n_frames))

    def n_in_m_bins(n, m):
        d, r = divmod(n, m)
        res = [[m * j + i for j in range(d)] for i in range(m)]
        for i in range(r):
            res[i].append(res[i][-1] + m)
        return res

    dts_array = n_in_m_bins(n_frames, n_proc)

    def calc_row(waterfall, dts):
        for j in dts:
            ab[j:, j] = (
                np.dot(waterfall[j:], waterfall[j]) / waterfall.shape[1]
            )
            ab[j, j:] = np.transpose(ab[j:, j])

    Parallel(
        n_jobs=n_proc, verbose=verbose, backend='threading'
    )(
        delayed(calc_row)(waterfall, dts_array[j]) for j in range(0, n_proc)
    )

    autocorrelation = (ab - I_bar_t1_I_bar_t2) / variance_squared
    return autocorrelation


def normalize(data: np.array):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def wfl(x):
    return x.reshape((x.shape[0], -1))


def extract_images_to_numpy_array(directory_path):
    """
    Extracts all image files from a specified directory, sorts them alphabetic-
    ally, inspects their original data, scales them to 8-bit grayscale (0-255),
    and stores their data in a T x M x N NumPy array.

    Args:
        directory_path (str): The path to the directory containing the images.

    Returns:
        numpy.ndarray or None: A NumPy array of shape (T, M, N) where T is the
                                number of images, M is the height, and N is the
                                width. Returns None if no images are found or
                                an error occurs.
    """
    image_data_list = []
    image_filenames = []

    image_extensions = (
        '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'
    )

    try:
        all_files = os.listdir(directory_path)
        for filename in all_files:
            if filename.lower().endswith(image_extensions):
                image_filenames.append(filename)

        image_filenames.sort()

        if not image_filenames:
            print(f"No image files found in directory: {directory_path}")
            return None

        print(f"Found {len(image_filenames)} image files. Processing...")

        expected_shape = None

        for filename in image_filenames:
            file_path = os.path.join(directory_path, filename)
            try:
                img_pil = Image.open(file_path)

                # Convert PIL Image to NumPy array to get raw pixel data
                img_array_raw = np.array(img_pil)

                print(f"Processing: {filename}")
                print(
                    f"  Original - Mode: {img_pil.mode},"
                    f"Dtype: {img_array_raw.dtype}, "
                    f"Shape: {img_array_raw.shape}, Min: {img_array_raw.min()}"
                    f", Max: {img_array_raw.max()}"
                )

                # --- Ensure image is grayscale and 2D ---
                # If image is RGB or RGBA, convert to grayscale ('L' mode in
                # Pillow)
                if img_pil.mode == 'RGB' or img_pil.mode == 'RGBA':
                    print(
                        f"  Converting {filename} from {img_pil.mode} to"
                        "grayscale ('L')..."
                    )
                    img_pil_gray = img_pil.convert('L')
                    img_array_processed = np.array(img_pil_gray)
                    print(
                        f"  Converted - Mode: {img_pil_gray.mode},"
                        f"Dtype: {img_array_processed.dtype}, "
                        f"Shape: {img_array_processed.shape}, Min:"
                        f"{img_array_processed.min()}, Max: "
                        f"{img_array_processed.max()}"
                    )
                elif img_array_raw.ndim == 3 and img_array_raw.shape[
                        2] == 1:  # e.g. (M, N, 1)
                    print(
                        f"  Squeezing single channel from shape"
                        f"{img_array_raw.shape}..."
                    )
                    img_array_processed = img_array_raw.squeeze(axis=2)
                elif img_array_raw.ndim == 2:  # Already 2D (likely grayscale)
                    img_array_processed = img_array_raw
                else:
                    print(
                        f"  Error: Image {filename} (mode: {img_pil.mode}, "
                        f"shape: {img_array_raw.shape}) "
                        "is not easily convertible to 2D grayscale. Skipping."
                    )
                    continue

                # --- Scale pixel values to 0-255 (uint8) ---
                # This is important for high bit-depth images
                # (e.g., uint16, float32)
                current_dtype = img_array_processed.dtype
                min_val = img_array_processed.min()
                max_val = img_array_processed.max()

                if (current_dtype == np.uint8 and min_val >= 0
                        and max_val <= 255):
                    # Already 8-bit and in range, no scaling needed
                    img_array_final = img_array_processed
                    print(f"  Image {filename} is already 8-bit grayscale.")
                elif max_val == min_val:
                    # Handle images with no dynamic range
                    # (flat color/intensity)
                    img_array_final = np.full(
                        img_array_processed.shape, 0, dtype=np.uint8
                    )
                    print(
                        f"  Image {filename} has no dynamic range (all pixels"
                        f"value {min_val}). Scaled to 0."
                    )
                else:
                    # Scale other types (uint16, int32, float32, etc.) or uint8
                    # outside 0-255 to 0-255 uint8
                    print(
                        f"  Scaling {filename} from dtype {current_dtype}"
                        f"(min: {min_val}, max: {max_val}) to uint8 (0-255)..."
                    )
                    # Ensure calculations are done with floating point numbers
                    # before converting to uint8
                    img_array_final = (
                        (img_array_processed.astype(np.float64) - min_val) /
                        (max_val - min_val) * 255.0
                    ).astype(np.uint8)

                print(
                    f"  Scaled - Dtype: {img_array_final.dtype}, "
                    f"Shape: {img_array_final.shape}, "
                    f"Min: {img_array_final.min()}, "
                    f"Max: {img_array_final.max()}"
                )

                # --- Shape check and append ---
                if expected_shape is None:
                    expected_shape = img_array_final.shape
                    print(
                        f"  Setting expected image dimensions "
                        "(Height x Width) "
                        f"to: {expected_shape}"
                    )
                elif img_array_final.shape != expected_shape:
                    print(
                        f"  Warning: Image {filename} has final dimensions"
                        f" {img_array_final.shape}, "
                        f"but expected {expected_shape}. Skipping this image."
                    )
                    continue

                image_data_list.append(img_array_final)

            except FileNotFoundError:
                print(f"  Error: File not found at {file_path}. Skipping.")
            except Exception as e:
                print(f"  Error processing image {filename}: {e}. Skipping.")

        if not image_data_list:
            print("No images were successfully processed.")
            return None

        final_numpy_array = np.stack(image_data_list, axis=0)

        print(
            "\nSuccessfully created NumPy array with shape:"
            f" {final_numpy_array.shape}"
        )
        return final_numpy_array

    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# TODO
# This was the previous filepath. Is this really the data that Nolan analyzed?
# image_array = extract_images_to_numpy_array(r"Skyrmion_Time_Series_Data")
# Bryan changed it to this:
mem('before image_array')
image_array = tifffile.imread(
    '../data/Skyrmion_Time_Series_300G_15sframe.tif'
)
mem('after image_array')

# ### LTEM Sim

# ### Sim Post-Processing

# You may need to reshape the data

# In[ ]:

N = 256
M = 48

tmp_img_array = image_array
del image_array
for i in range(len(tmp_img_array)):
    tmp_img_array[i] = tmp_img_array[i] / np.mean(tmp_img_array[i])
tmp_img_array.shape
mem('after tmp_image_array')

# In[ ]:

# Bryan removed this
# tmp_img_array = tmp_img_array[36:36 + 570]
# tmp_img_array = tmp_img_array[:, :128, :128]

# ## Window 2-TCF

# In[49]:


# window to grab parts of an array
def window(arr, x, y, size):
    return arr[:,
               np.maximum(x - size
                          // 2, 0):np.minimum(x + size // 2, arr.shape[1]),
               np.maximum(y - size
                          // 2, 0):np.minimum(y + size // 2, arr.shape[2])]


# Bryan removed this too
# rand_arr = np.random.randint(0, 32, (64, 32, 32))
# print(np.std(rand_arr[:, 18:22, 26:30] - window(rand_arr, 20, 28, 4)))
# window(rand_arr, 4, 4, 8).shape

# In[50]:
# expecting a T x M x N array, where T is time axis, (M,N) are picture
# dimensions
data_source = tmp_img_array
n_frames = data_source.shape[0]
# TODO
# Check this is the right window size??
# Bryan commented out window_size and window_step_size in lieu of command line parsing (see above)
# window_size = 50
# window_step_size = 3


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)





mem('before window coords')
window_coords = np.meshgrid(
    np.arange(
        window_size // 2 + 1, data_source.shape[1] - window_size // 2 - 1,
        window_step_size
    ),
    np.arange(
        window_size // 2 + 1, data_source.shape[2] - window_size // 2 - 1,
        window_step_size
    )
)
mem('after window coords')
mem('before import vecs')
with open(feature_vecs_path, 'rb') as fh:
    scaled_feature_vecs = np.load(fh)
mem('after import vecs')

# --- 1. DEFINE YOUR PARAMETERS ---
# The side length of each square box in data units
sidelength = window_step_size
box_transparency = 0.875  # The alpha value for the boxes

mem('before original coords')
grid_x, grid_y = window_coords
original_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
mem('after original coords')

# Create dummy high-dimensional data
np.random.seed(42)

# --- 3. THE ANALYSIS PIPELINE (SCALE AND REDUCE) ---

mem('before umap')
# Run UMAP to get the 2D embedding
# Bryan added the umap / pca choice and the pca
if mapper_str == 'umap':
    mapper = umap.UMAP(
        n_components=3, n_neighbors=15, min_dist=0.1, random_state=42
    )
    umap_embedding = mapper.fit_transform(scaled_feature_vecs)
elif mapper_str == 'pca':
    mapper = skPCA(n_components=3)
    mapper.fit(scaled_feature_vecs)
    umap_embedding = mapper.transform(scaled_feature_vecs)
# Bryan added this
del scaled_feature_vecs
mem('after umap')

# --- 4. MAP UMAP COORDINATES TO EXPLICIT RGB COLORS ---
print("Mapping UMAP coordinates to 3-channel RGB colors...")
# Normalize the UMAP dimensions to be in the [0, 1] range for color values


def norm(val):
    return (val - val.min()) / (val.max() - val.min())


def norm_dim(idx):
    return norm(umap_embedding[:, idx - 1])


# Create an (N, 3) array where N is the number of points.
# Each row is a specific (Red, Green, Blue) triplet.
mem('before umap colors')
rgb_colors = np.zeros((len(original_coords), 3))
rgb_colors[:, 0] = norm_dim(1)  # Map UMAP Dimension 1 to the Red channel
rgb_colors[:, 1] = norm_dim(2)  # Map UMAP Dimension 2 to the Green channel
rgb_colors[:, 2] = norm_dim(3)
mem('after umap colors')

# --- 5. CREATE A LIST OF RECTANGLE PATCHES ---
print("Creating rectangle patches...")
patches = []
# Assuming 'original_coords' are the bottom-left corners of the boxes
mem('before rectangles')
for coord in tqdm(original_coords):
    rect = Rectangle((coord[1], coord[0]), sidelength, sidelength)
    patches.append(rect)
mem('after rectangles')

# --- 6. CREATE AND ADD THE PATCH COLLECTION ---
print("Creating and adding patch collection to plot...")
# When using explicit RGB colors, we pass them to the 'facecolor' argument.
# We do NOT use 'cmap' or 'set_array'.
mem('before collection')
collection = PatchCollection(
    patches,
    alpha=box_transparency,
    facecolor=rgb_colors,  # Use the explicit (N, 3) color array
    edgecolor='none'  # Turn off box edges for a smoother look
)
mem('after collection')

# --- 7. CREATE THE PLOT ---
fig, ax = plt.subplots(figsize=(6, 5))

background = np.flip(tmp_img_array[0], axis=0)
img_height, img_width = background.shape[:2]
ax.imshow(background, extent=[0, img_width, 0, img_height])

ax.add_collection(collection)

# ADD ROI
mem('after collection')

# Set plot limits and aspect ratio
# ax.autoscale_view()
mem('before plotting')
ax.set_xlim([0, background.shape[1]])
ax.set_ylim([0, background.shape[0]])
ax.set_aspect('equal')

ax.set_title('Data with UMAP to RGB Overlay', fontsize=16)
ax.set_xlabel('x-axis (px)', fontsize=12)
ax.set_ylabel('y-axis (px)', fontsize=12)
plt.grid(False)
plt.savefig(output_file)
mem('after plotting')
