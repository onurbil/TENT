import os
from pathlib import Path

"""
This file contains all paths necessary for tensorized_transformers:

An example to import:
from paths import WORKING_DIR, PARENT_WORKING_DIR, DATASET_DIR
"""

# Path of this file:
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path of main.py as main working directory:
WORKING_DIR =  Path(THIS_FILE_DIR).parent
# Parent of main.py to save datasets:
PARENT_WORKING_DIR = Path(WORKING_DIR).parent
# Raw dataset directory for preprocessing:
DATASET_DIR = os.path.join(PARENT_WORKING_DIR, 'dataset')
# Path of zip file of the raw dataset:
ZIP_PATH = os.path.join(DATASET_DIR,'historical_hourly_weather_data.zip')
# Preprocessed dataset directory:
PROCESSED_DATASET_DIR = os.path.join(PARENT_WORKING_DIR, 'processed_dataset')
