import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from common.paths import WORKING_DIR, PARENT_WORKING_DIR, DATASET_DIR, PROCESSED_DATASET_DIR


"""
Encode the categories in the weather_description.csv:

To load the array please use:
np.load('path to file', allow_pickle=True)
"""


def encode_weather_desc(input_folder, output_folder, file, norm=True):
    """
    Encode the categories in [file]
    norm: (bool) Normalize dataset between [0,1]
    """
    file_path = os.path.join(input_folder, file)
    df_table = pd.read_csv(file_path)
    df_table.fillna(method='ffill', inplace=True)
    df_table.fillna(method='bfill', inplace=True)
    df_table.iloc[:,1:] = df_table.iloc[:,1:].apply(LabelEncoder().fit_transform)
    np_table = df_table.to_numpy()
    if norm:
        np_table[:,1:]=np_table[:,1:]/np_table[:,1:].max()

    new_filename = Path(file).stem
    new_filepath = os.path.join(output_folder, new_filename)
    np.save(new_filepath, np_table)


def main():
    
    if not os.path.exists(PROCESSED_DATASET_DIR):
        os.makedirs(PROCESSED_DATASET_DIR)
    
    file = 'weather_description.csv'
            
    encode_weather_desc(DATASET_DIR, PROCESSED_DATASET_DIR, file)


if __name__ == "__main__":
    main()
