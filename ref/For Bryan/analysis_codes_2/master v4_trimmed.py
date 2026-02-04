#!/usr/bin/env python
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

# XPCS packages

from PyTTCF.utils.ttcf import *
from Hao_speckle_utils import *

# Imaging / Plotting

from PIL import Image  # Pillow library for image manipulation
import matplotlib as mpl
import matplotlib.pyplot as plt

import os

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

os.environ["OMP_NUM_THREADS"] = "64"

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


def normalize(data: np.array):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# In[54]:

# In[55]:


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


image_array = extract_images_to_numpy_array(r"Skyrmion_Time_Series_Data")

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

# In[ ]:

tmp_img_array = tmp_img_array[36:36 + 570]

# ## Window 2-TCF

# In[49]:


# window to grab parts of an array
def window(arr, x, y, size):
    return arr[:,
               np.maximum(x - size
                          // 2, 0):np.minimum(x + size // 2, arr.shape[1]),
               np.maximum(y - size
                          // 2, 0):np.minimum(y + size // 2, arr.shape[2])]


rand_arr = np.random.randint(0, 32, (64, 32, 32))
print(np.std(rand_arr[:, 18:22, 26:30] - window(rand_arr, 20, 28, 4)))
window(rand_arr, 4, 4, 8).shape

# In[50]:
# expecting a T x M x N array, where T is time axis, (M,N) are picture
# dimensions
data_source = tmp_img_array
n_frames = data_source.shape[0]
window_size = 50
window_step_size = 3


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


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
window_ttcf = np.zeros(
    (window_coords[0].shape[0], window_coords[0].shape[1], n_frames, n_frames)
)

# In[51]:

window_ttcf.shape

# ### Windowed autocorrelation
# Calculates autocorrelation 2-TCF for each window so that they can be
# classified on a pointwise basis

# In[57]:
"""
for i in tqdm(range(window_coords[0].shape[0])):
    for j in range(window_coords[0].shape[1]):
        target_window = window(data_source, \
                               window_coords[0][i][j], \
                               window_coords[1][i][j], \
                               window_size)
        #print(target_window.shape)
        waterfall = wfl(target_window)
        #print(waterfall.shape)
        tt = calc_autocorrelation(waterfall)
        window_ttcf[i,j,:,:] = tt
"""


def add_to_window_ttcf(pair):
    i, j = pair
    target_window = window(
        data_source,
        window_coords[0][i][j],
        window_coords[1][i][j],
        window_size,
    )
    waterfall = wfl(target_window)
    tt = autocorr(waterfall)
    return i, j, tt


i_values = np.arange(0, window_ttcf.shape[0])
j_values = np.arange(0, window_ttcf.shape[1])
I, J = np.meshgrid(i_values, j_values, indexing='ij')
pairs = np.stack((I.ravel(), J.ravel()), axis=-1)
print(len(pairs))

results = Parallel(
    n_jobs=16, backend='loky'
)(
    delayed(add_to_window_ttcf)(pair)
    for pair in tqdm(pairs, desc="Processing 2-TCF")
)

print("Consolidating results...")
for i, j, tt_data in tqdm(results):
    window_ttcf[i, j, :, :] = tt_data

# ### Window correlation to ROI window
# Defines a window considered ROI and generates correlation 2-TCF with all
# other windows. For analysis to determine how similar behavior is?

# In[ ]:

# WIP

# ### Window causation?

# ### Neural Network Classification

# In[33]:

window_ttcf = window_ttcf[:, :, 920:, 920:]

# In[31]:

plt.imshow(window_ttcf[4, 4])
plt.clim([-1, 1])
plt.colorbar()

# In[63]:

# Instantiate pre-trained model

weights = EfficientNet_B2_Weights.DEFAULT  # Loads the best available weights
model = efficientnet_b2(weights=weights)
model.eval()
print(model)

# Strip away classifier layer so we just get feature vectors

model.classifier = torch.nn.Identity()

# Reformate data to have 3 channels so that
# EfficientNet can properly process the data

A, B, C, D = window_ttcf.shape
num_items = A * B
reshaped_array = window_ttcf.reshape(num_items, C, D)

img_size = weights.transforms().crop_size  # Typically [224] for B0
norm_mean = weights.transforms().mean  # e.g., [0.485, 0.456, 0.406]
norm_std = weights.transforms().std  # e.g., [0.229, 0.224, 0.225]


def preprocess_numpy_batch(np_batch):
    """
    Normalizes and prepares a 3D NumPy array (a batch of intensity fields)
    for a 3-channel model.
    """
    processed_tensors = []
    for single_np_array in np_batch:
        single_np_array = single_np_array.astype(np.float32)

        min_val, max_val = single_np_array.min(), single_np_array.max()
        if max_val > min_val:
            single_np_array = (single_np_array - min_val) / (max_val - min_val)
        else:
            single_np_array = np.zeros_like(single_np_array)

        tensor = torch.from_numpy(single_np_array)
        tensor = F.resize(tensor.unsqueeze(0), size=img_size, antialias=True)
        tensor = tensor.repeat(3, 1, 1)
        tensor = F.normalize(tensor, mean=norm_mean, std=norm_std)
        processed_tensors.append(tensor)

    return torch.stack(processed_tensors, dim=0)


net_input = torch.tensor(reshaped_array)
net_input = net_input.unsqueeze(1)
net_input = torch.tile(net_input, (1, 3, 1, 1)).type(torch.float32)

# In[64]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)
model.eval()

# Batching

batch_size = 512
all_feature_vectors = []

with torch.no_grad():
    print(f"Processing {len(net_input)} images in batches of {batch_size}...")

    for i in range(0, len(net_input), batch_size):
        # Define the end of the batch
        end = i + batch_size

        # Critically, we only move the small batch to the GPU ---
        batch = net_input[i:end]
        batch_gpu = batch.to(device)
        output_embeddings = model(batch_gpu)
        all_feature_vectors.append(output_embeddings.cpu())

feature_vectors_tensor = torch.cat(all_feature_vectors, dim=0)
feature_vectors = feature_vectors_tensor.numpy()

print("Scaling features...")
scaler = StandardScaler()
scaled_feature_vecs = scaler.fit_transform(feature_vectors)
del net_input

# --- 1. DEFINE YOUR PARAMETERS ---
# The side length of each square box in data units
sidelength = window_step_size
box_transparency = 0.875  # The alpha value for the boxes

grid_x, grid_y = window_coords
original_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

# Create dummy high-dimensional data
np.random.seed(42)

# --- 3. THE ANALYSIS PIPELINE (SCALE AND REDUCE) ---

# Run UMAP to get the 2D embedding
mapper = umap.UMAP(
    n_components=3, n_neighbors=15, min_dist=0.1, random_state=42
)
umap_embedding = mapper.fit_transform(scaled_feature_vecs)

# --- 4. MAP UMAP COORDINATES TO EXPLICIT RGB COLORS ---
print("Mapping UMAP coordinates to 3-channel RGB colors...")
# Normalize the UMAP dimensions to be in the [0, 1] range for color values


def norm(val):
    return (val - val.min()) / (val.max() - val.min())


def norm_dim(idx):
    return norm(umap_embedding[:, idx - 1])


# Create an (N, 3) array where N is the number of points.
# Each row is a specific (Red, Green, Blue) triplet.
rgb_colors = np.zeros((len(original_coords), 3))
rgb_colors[:, 0] = norm_dim(1)  # Map UMAP Dimension 1 to the Red channel
rgb_colors[:, 1] = norm_dim(2)  # Map UMAP Dimension 2 to the Green channel
rgb_colors[:, 2] = norm_dim(3)

# --- 5. CREATE A LIST OF RECTANGLE PATCHES ---
print("Creating rectangle patches...")
patches = []
# Assuming 'original_coords' are the bottom-left corners of the boxes
for coord in tqdm(original_coords):
    rect = Rectangle((coord[1], coord[0]), sidelength, sidelength)
    patches.append(rect)

# --- 6. CREATE AND ADD THE PATCH COLLECTION ---
print("Creating and adding patch collection to plot...")
# When using explicit RGB colors, we pass them to the 'facecolor' argument.
# We do NOT use 'cmap' or 'set_array'.
collection = PatchCollection(
    patches,
    alpha=box_transparency,
    facecolor=rgb_colors,  # Use the explicit (N, 3) color array
    edgecolor='none'  # Turn off box edges for a smoother look
)

# --- 7. CREATE THE PLOT ---
fig, ax = plt.subplots(figsize=(6, 5))

background = np.flip(tmp_img_array[0], axis=0)
img_height, img_width = background.shape[:2]
ax.imshow(background, extent=[0, img_width, 0, img_height])

ax.add_collection(collection)

# ADD ROI


def get_clustered_indices(colors, m):
    """
    Finds the indices of M closely clustered colors from an Nx3 RGB array.

    Args:
        colors (np.ndarray): An Nx3 NumPy array of RGB colors.
        m (int): The number of colors/indices to sample.

    Returns:
        np.ndarray: An array of M indices corresponding to the most
                    clustered colors in the original 'colors' array.
    """
    if m > len(colors):
        raise ValueError("M cannot be larger than the number of colors N.")

    # Calculate the pairwise Euclidean distances between all colors
    pairwise_distances = np.linalg.norm(
        colors[:, np.newaxis, :] - colors[np.newaxis, :, :], axis=2
    )

    # Find the index of the color with the minimum average distance to all
    # others. This color serves as the cluster's "medoid" or center.
    avg_distances = np.mean(pairwise_distances, axis=1)
    center_index = np.argmin(avg_distances)

    # Get the distances from this center color to all other colors
    distances_from_center = pairwise_distances[center_index]

    # Find the indices of the M closest colors
    closest_indices = np.argsort(distances_from_center)[:m]

    return closest_indices


# Get the indices of the M most clustered colors
clustered_indices = get_clustered_indices(rgb_colors, 12)

# Use these indices to retrieve the original 2D coordinates
sampled_coords = original_coords[clustered_indices]

# You can also get the actual color values if needed for verification
sampled_colors = rgb_colors[clustered_indices]

# Set plot limits and aspect ratio
# ax.autoscale_view()
ax.set_xlim([0, background.shape[1]])
ax.set_ylim([0, background.shape[0]])
ax.set_aspect('equal')

ax.set_title('Data with UMAP to RGB Overlay', fontsize=16)
ax.set_xlabel('x-axis (px)', fontsize=12)
ax.set_ylabel('y-axis (px)', fontsize=12)
plt.grid(False)
plt.show()
