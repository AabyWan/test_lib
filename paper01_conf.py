# Things that are useful across all steps in the framework.
import os, pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Get the working directory according to where the .py file is executed
SCRIPT_DIR = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1])
#SCRIPT_DIR = f"C:/Users/aabywan/Downloads/Flickr_8k"

# Make sure SCRIPT_DIR is the current working dir.
os.chdir(SCRIPT_DIR)

# Make folder for outputs if not already there.
pathlib.Path("./demo_outputs/figs").mkdir(parents=True, exist_ok=True)

IMGPATH = os.path.join(SCRIPT_DIR, "Images")

# Set a uniform imagesize for all figures
FIGSIZE    = (5, 3)

# Could set m_dict here too ?!
M_DICT = {"Hamming": "hamming", "Cosine": "cosine"}