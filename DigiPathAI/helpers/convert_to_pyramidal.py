import os, sys
import numpy as np
import json
import glob
import time

def create_pyramidal_img(img_path, output_image_path):
    """ Convert normal image to pyramidal image.
    Parameters
    -------
    img_path: str
        Absolute path of Whole slide image path (absolute path is needed)
    output_image_path: str
        Absolute path of of the saved the generated pyramidal image with extension tiff,

    Returns
    -------
    status: int
        The status of the pyramidal image generation (0 stands for success)
    Notes
    -------
    ImageMagick need to be preinstalled to use this function.
    >>> sudo apt-get install imagemagick
    Examples
    --------
    >>> img_path = os.path.join(PRJ_PATH, "test/data/Images/CropBreastSlide.tif")
    >>> save_dir = os.path.join(PRJ_PATH, "test/data/Slides")
    >>> status = pyramid.create_pyramidal_img(img_path, save_dir)
    >>> assert status == 0
    """

    convert_cmd = "convert " + os.path.abspath(img_path)
    convert_option = " -compress LZW -define tiff:tile-geometry=256x256 ptif:"
    img_name = os.path.basename(img_path)
    # convert_dst = os.path.join(save_dir, os.path.splitext(img_name)[0] + ".tiff")
    convert_dst = output_image_path
    status = os.system(convert_cmd + convert_option + convert_dst)

    return status

