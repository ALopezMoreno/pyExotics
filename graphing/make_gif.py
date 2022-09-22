import glob
from PIL import Image
import os

# filepaths
fp_in = "/home/andres/Desktop/pyExotics/movieHist/*.png"
fp_out = "/home/andres/Desktop/pyExotics/gifs/s14_s24_p24_evolution_smooth.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
imgs = (Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getmtime))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=75, loop=0)
