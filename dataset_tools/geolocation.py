import os
import numpy as np
import pandas as pd
from pathlib import Path
from common.paths import WORKING_DIR, PARENT_WORKING_DIR, DATASET_DIR, PROCESSED_DATASET_DIR
from common.variables import DATASET_SAMPLE_SIZE

"""
Convert the latitude, longitude of cities in city_attributes.csv to x,y,z
coordinates:

To load the array please use:
np.load('path to file', allow_pickle=True)
"""

def convert_to_xyz(lat, lon):
    """
    Convert latitude, longitude to x,y,z coordinates:
    lat: latitude
    lon: longitude
    """
    lat = np.array(lat, dtype=np.float32)
    lon = np.array(lon, dtype=np.float32)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x,y,z


def normalize_xyz(x,y,z):
    """
    convert_to_xyz() function normalizes to (-1,1). This function normalizes
    after the output of convert_to_xyz() to (0,1):
    """
    norm_x = (x+1)/2
    norm_y = (y+1)/2
    norm_z = (z+1)/2

    return norm_x, norm_y, norm_z


def location_xyz(input_folder, output_folder, file, repeat=45253, norm=True):
    """
    Convert the latitude, longitude of cities in city_attributes.csv to x,y,z
    coordinates:
    norm: (bool) Normalize dataset between [0,1].
    """
    file_path = os.path.join(input_folder, file)
    df_table = pd.read_csv(file_path)
    np_table = df_table.to_numpy()

    latitude = np_table[:,2]
    longitude = np_table[:,3]
    x,y,z = convert_to_xyz(latitude, longitude)

    if norm:
        norm_x, norm_y, norm_z = normalize_xyz(x,y,z)

    coordinates = np.empty((repeat,np_table.shape[0],0))
    for coord in [norm_x, norm_y, norm_z]:

        locations = np.expand_dims(coord, axis=(0,2))
        locations = np.tile(locations, (repeat,1,1))
        coordinates = np.append(coordinates, locations, axis=2)

    new_filename = Path(file).stem
    new_filepath = os.path.join(output_folder, new_filename)
    np.save(new_filepath, coordinates)

    return coordinates


def main():

    if not os.path.exists(PROCESSED_DATASET_DIR):
        os.makedirs(PROCESSED_DATASET_DIR)

    file = 'city_attributes.csv'
    coordinates = location_xyz(DATASET_DIR, PROCESSED_DATASET_DIR, file,
                               repeat=DATASET_SAMPLE_SIZE, norm=True)


if __name__ == "__main__":
    main()
