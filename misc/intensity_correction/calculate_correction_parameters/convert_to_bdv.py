from pybdv import make_bdv
from skimage.io import imread
import os


def convert_to_bdv(correction_folder, name):
    raw = imread(os.path.join(correction_folder, name, 'corrected_raw.tiff'))

    make_bdv(raw, output_path=os.path.join(correction_folder, name, str(name) + '.h5'),
             resolution=[0.025, 0.32, 0.32], unit='micrometer')
