import os
import numpy as np
import pandas as pd
from pathlib import Path
from common.paths import WORKING_DIR, PARENT_WORKING_DIR, DATASET_DIR, PROCESSED_DATASET_DIR
from common.variables import weather_desc_labels, TRAIN_VAL_SIZE


"""
Encode the categories in the weather_description.csv:

To load the array please use:
np.load('path to file', allow_pickle=True)

Categories:

['shower drizzle', 'freezing rain', 'volcanic ash', 'proximity shower rain', 'fog', 'shower snow', 'tornado', 'drizzle', 'heavy shower snow', 'few clouds', 'proximity sand/dust whirls', 'mist', 'light rain', 'light shower sleet', 'rain and snow', 'proximity thunderstorm with rain', 'thunderstorm with heavy drizzle', 'overcast clouds', 'sky is clear', 'light rain and snow', 'proximity moderate rain', 'light intensity drizzle rain', 'heavy thunderstorm', 'thunderstorm with rain', 'scattered clouds', 'sand/dust whirls', 'moderate rain', 'broken clouds', 'shower rain', 'smoke', 'haze', 'heavy intensity shower rain', 'sleet', 'squalls', 'heavy snow', 'sand', 'ragged shower rain', 'thunderstorm with heavy rain', 'ragged thunderstorm', 'thunderstorm with light rain', 'thunderstorm with light drizzle', 'light intensity shower rain', 'snow', 'heavy intensity rain', 'light shower snow', 'thunderstorm with drizzle', 'heavy intensity drizzle', 'thunderstorm', 'light snow', 'proximity thunderstorm', 'light intensity drizzle', 'dust', 'proximity thunderstorm with drizzle', 'very heavy rain']
"""


def encode_weather_desc(input_folder, output_folder, file, train_val, norm=True):
    """
    Encode the categories in [file]
    norm: (bool) Normalize dataset between [0,1]
    """
    file_path = os.path.join(input_folder, file)

    df_table = pd.read_csv(file_path)
    df_table.fillna(method='ffill', inplace=True)
    df_table.fillna(method='bfill', inplace=True)

    for i in list(df_table.columns)[1:]:
        df_table[i] = df_table[i].map(weather_desc_labels)

    np_table = df_table.to_numpy()
    norm_table = np_table[:train_val]

    if norm:
        max_val = norm_table[:,1:].max()
        min_val = norm_table[:,1:].min()
        np_table[:,1:] = (np_table[:,1:] - min_val)/(max_val - min_val)


    new_filename = Path(file).stem
    new_filepath = os.path.join(output_folder, new_filename)
    np.save(new_filepath, np_table)

    new_scale_filepath = os.path.join(output_folder, 'scale.npy')
    scale = [[file, min_val, max_val]]
    np.save(new_scale_filepath, scale)


def main():

    if not os.path.exists(PROCESSED_DATASET_DIR):
        os.makedirs(PROCESSED_DATASET_DIR)

    file = 'weather_description.csv'

    encode_weather_desc(DATASET_DIR, PROCESSED_DATASET_DIR, file, TRAIN_VAL_SIZE)


if __name__ == "__main__":
    main()
