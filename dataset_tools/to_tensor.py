import os
import numpy as np
import pandas as pd
from pathlib import Path
from common.paths import PROCESSED_DATASET_DIR

"""
Creates a tensor from all datasets with name 'dataset_tensor.npy':
"""

def dataset_to_tensor(input_folder, output_folder, files):
    """
    Create from the arrays in the 'processed dataset' folder a tensor with 
    shape (6,45253,37) with name 'dataset_tensor.npy'.
    """
    dataset_tensor = np.empty([0,45253,37])
    for file in files:
        
        file_path = os.path.join(input_folder, file)
        np_table = np.load(file_path, allow_pickle=True)
        np_table = np.expand_dims(np_table, axis=0)    
        dataset_tensor = np.concatenate((dataset_tensor,np_table), axis=0)
    new_filename = 'dataset_tensor'
    new_filepath = os.path.join(output_folder, new_filename)
    np.save(new_filepath, dataset_tensor)


def main():
    
    files = ['humidity.npy', 'pressure.npy', 'temperature.npy', 
             'weather_description.npy', 'wind_direction.npy', 
             'wind_speed.npy']

    dataset_to_tensor(PROCESSED_DATASET_DIR, PROCESSED_DATASET_DIR, files)
    print("Dataset preprocessed...")


if __name__ == "__main__":
    main()  