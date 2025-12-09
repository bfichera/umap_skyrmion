import os
from pathlib import Path

from tifffile import imread

data_directory = Path(os.environ['UMAP_SKYRMION_DATA'])


def load(*args, **kwargs):
    with open(data_directory / 'Skyrmion_Time_Series_300G_15sframe.tif',
              'rb') as fh:
        return imread(fh)
