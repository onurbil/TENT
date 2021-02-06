import os
import numpy as np
import pandas as pd
from pathlib import Path
from common.paths import PROCESSED_DATASET_DIR


"""
Creates a tensor from all datasets with name 'dataset_tensor.npy':
"""

def process_timestamp(date):

    data_list = [pd.to_datetime(pd.Series(x)) for x in date]
    # Transforming datetime to day of the year
    doy = pd.DataFrame([x.apply(lambda x: x.timetuple().tm_yday)/366 for x in data_list]).values
    doy = np.expand_dims(doy, 1)
    # Transforming datetime to time of the day
    hod = pd.DataFrame([x.apply(lambda x: x.timetuple().tm_hour)/23 for x in data_list]).values
    hod = np.expand_dims(hod, 1)
    return doy, hod

def dataset_to_tensor(input_folder, output_folder, files):
    """
    Create from the arrays in the 'processed dataset' folder a tensor with
    shape (45253,36,9) and name 'dataset_tensor.npy' and cleans the time
    stamps in the rows.
    """
    dataset_tensor = np.empty([45253,37,0])
    for file in files[:-1]:

        file_path = os.path.join(input_folder, file)
        np_table = np.load(file_path, allow_pickle=True)
        np_table = np.expand_dims(np_table, axis=-1)
        dataset_tensor = np.concatenate((dataset_tensor,np_table), axis=-1)

    # Process the timestamp
    doy, hod = process_timestamp(dataset_tensor[:, 0, 0])

    # Clean the time stamp:
    dataset_tensor = dataset_tensor[:,1:,:]

    # Add city attributes:
    city_att_path = os.path.join(input_folder, files[-1])
    city_att = np.load(city_att_path, allow_pickle=True)
    dataset_tensor = np.concatenate((dataset_tensor, city_att), axis=-1)
    # Add hour of the day
    hod = np.broadcast_to(hod, (dataset_tensor.shape[0], dataset_tensor.shape[1], 1))
    dataset_tensor = np.concatenate((hod, dataset_tensor), axis=-1)
    # Add day of the year
    doy = np.broadcast_to(doy, (dataset_tensor.shape[0], dataset_tensor.shape[1], 1))
    dataset_tensor = np.concatenate((doy, dataset_tensor), axis=-1)

    # Save dataset_tensor:
    new_filename = 'dataset_tensor'
    new_filepath = os.path.join(output_folder, new_filename)
    np.save(new_filepath, dataset_tensor)

def main():

    files = ['humidity.npy', 'pressure.npy', 'temperature.npy',
             'weather_description.npy', 'wind_direction.npy',
             'wind_speed.npy', 'city_attributes.npy']

    dataset_to_tensor(PROCESSED_DATASET_DIR, PROCESSED_DATASET_DIR, files)
    print("Dataset preprocessed...")

if __name__ == "__main__":
    main()
