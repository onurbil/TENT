import os
import numpy as np
import pandas as pd
from common.paths import EU_DATASET_DIR, EU_PROCESSED_DATASET_DIR
from common.variables import EU_TRAIN_VAL_SIZE



def eu_process(input_folder, output_folder, files, train_val):


    np_table = []
    for file in files:

        file_path = os.path.join(input_folder, file)
        df_table = pd.read_csv(file_path)
        features = df_table.columns[1:]
        table = df_table.to_numpy()
        np_table.append(table)


    np_table = np.moveaxis(np.array(np_table),0,1)[:,:,1:]
    # np_table: time x cities x features:

    norm_table = np_table[:train_val]

    scale_list = []
    for feature in range(len(features)):

        max_val = norm_table[:,:, feature].max()
        min_val = norm_table[:,:, feature].min()
        np_table[:,:,feature] = (np_table[:,:,feature] - min_val)/(max_val - min_val)

        scale_list.append([features[feature], min_val, max_val])

    scale_list = np.array(scale_list)

    filename = 'eu_dataset_tensor.npy'
    output_filepath = os.path.join(output_folder, filename)
    np.save(output_filepath, np_table)


    scale_filepath = os.path.join(output_folder, 'eu_scale.npy')
    np.save(scale_filepath, scale_list)


def main():

    if not os.path.exists(EU_PROCESSED_DATASET_DIR):
        os.makedirs(EU_PROCESSED_DATASET_DIR)

    files = ['amsterdam.csv', 'barcelona.csv', 'berlin.csv', 'brussels.csv', 'copenhagen.csv', 'dublin.csv', 'frankfurt.csv', 'hamburg.csv',
             'london.csv', 'luxembourg.csv', 'lyon.csv', 'maastricht.csv', 'malaga.csv', 'marseille.csv', 'munich.csv', 'nice.csv', 'paris.csv', 'rotterdam.csv',]

    eu_process(EU_DATASET_DIR, EU_PROCESSED_DATASET_DIR, files, EU_TRAIN_VAL_SIZE)
    print("Dataset preprocessed...")


if __name__ == "__main__":
    main()
