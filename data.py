from pathlib import Path

from tifffile import imread
import matplotlib.pyplot as plt


with open('data/Skyrmion_Time_Series_300G_15sframe.tif', 'rb') as fh:
    ims = imread(fh)

for i in range(ims.shape[0]):
    plt.imshow(ims[i, :, :])
    plt.show()
