import os
import numpy as np
import pandas as pd
from common.paths import EU_DATASET_DIR, EU_PROCESSED_DATASET_DIR
from common.variables import EU_TRAIN_VAL_SIZE, EU_DATASET_SAMPLE_SIZE


def eu_process_timestamp(date):

    data_list = [pd.to_datetime(pd.Series(x)) for x in date]
    # Transforming datetime to day of the year
    doy = pd.DataFrame([x.apply(lambda x: x.timetuple().tm_yday)/366 for x in data_list]).values  
    doy = np.expand_dims(doy, 1)

    return doy


def eu_convert_to_xyz(lat, lon):
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


def eu_normalize_xyz(x,y,z):
    """
    eu_convert_to_xyz() function normalizes to (-1,1). This function normalizes
    after the output of convert_to_xyz() to (0,1):
    """
    norm_x = (x+1)/2
    norm_y = (y+1)/2
    norm_z = (z+1)/2

    return norm_x, norm_y, norm_z


def eu_location_xyz(input_folder, file, repeat=45253, norm=True):
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
    x,y,z = eu_convert_to_xyz(latitude, longitude)

    if norm:
        norm_x, norm_y, norm_z = eu_normalize_xyz(x,y,z)

    coordinates = np.empty((repeat,np_table.shape[0],0))
    for coord in [norm_x, norm_y, norm_z]:

        locations = np.expand_dims(coord, axis=(0,2))
        locations = np.tile(locations, (repeat,1,1))
        coordinates = np.append(coordinates, locations, axis=2)

    return coordinates


def eu_process(input_folder, output_folder, files, train_val):

    np_table = []
    for file in files:

        file_path = os.path.join(input_folder, file)
        df_table = pd.read_csv(file_path)
        features = df_table.columns[1:]
        table = df_table.to_numpy()
        np_table.append(table)

    np_table = np.moveaxis(np.array(np_table),0,1)
    # np_table: time x cities x features
    
    # Encode day of the year:
    doy = eu_process_timestamp(np_table[:,0,0])
    # Remove 'date' column: 
    np_table = np_table[:,:,1:]    

    norm_table = np_table[:train_val]

    scale_list = []
    for feature in range(len(features)):

        max_val = norm_table[:,:, feature].max()
        min_val = norm_table[:,:, feature].min()
        np_table[:,:,feature] = (np_table[:,:,feature] - min_val)/(max_val - min_val)

        scale_list.append([features[feature], min_val, max_val])
    scale_list = np.array(scale_list)
    
    # Add day of the year (doy) to the dataset:
    doy = np.broadcast_to(doy, (np_table.shape[0], np_table.shape[1], 1))
    np_table = np.concatenate((doy, np_table), axis=-1)
    scale_list = np.append([['day_of_the_year', 0, 366]], scale_list, axis=0)

    # Add geolocation information:
    # city_att_file = 'city_attributes_eu.csv'
    # city_att = eu_location_xyz(input_folder, city_att_file, 
    #                         repeat=EU_DATASET_SAMPLE_SIZE, norm=True)    
    # np_table = np.concatenate((np_table, city_att), axis=-1)


        
    filename = 'eu_dataset_tensor.npy'
    output_filepath = os.path.join(output_folder, filename)
    np.save(output_filepath, np_table)


    scale_filepath = os.path.join(output_folder, 'eu_scale.npy')
    np.save(scale_filepath, scale_list)


def main():

    if not os.path.exists(EU_PROCESSED_DATASET_DIR):
        os.makedirs(EU_PROCESSED_DATASET_DIR)

    files = ['amsterdam.csv', 'barcelona.csv', 'berlin.csv', 'brussels.csv', 
             'copenhagen.csv', 'dublin.csv', 'frankfurt.csv', 'hamburg.csv',
             'london.csv', 'luxembourg.csv', 'lyon.csv', 'maastricht.csv', 
             'malaga.csv', 'marseille.csv', 'munich.csv', 'nice.csv', 
             'paris.csv', 'rotterdam.csv',]

    eu_process(EU_DATASET_DIR, EU_PROCESSED_DATASET_DIR, files, EU_TRAIN_VAL_SIZE)
    print("Dataset preprocessed...")


if __name__ == "__main__":
    main()
