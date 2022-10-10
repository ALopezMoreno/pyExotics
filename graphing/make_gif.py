import glob
from PIL import Image
import os

# filepaths
fp_in = "/home/andres/Desktop/pyExotics/movieTriangle/*.png"
fp_out = "/home/andres/Desktop/pyExotics/gifs/dcp_evolution.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
imgs = (Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getmtime))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=50, loop=0)
