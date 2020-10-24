import os
from zipfile import ZipFile
from tqdm import tqdm
import requests
from common.paths import WORKING_DIR, PARENT_WORKING_DIR, DATASET_DIR, ZIP_PATH


"""
This file checks and if necessary downloads and extracts the dataset according
to the folder structure.

Folder structure:
For reading the dataset and creating the models the folder structure must be as 
following:
<folder_name> is the parent folder of github repository and may have any name.

<folder_name>
└─── tensorized_transformers/
│    └─── main.py
│    └─── ...
│
└─── dataset/
│    └─── city_attributes.csv
│    └─── humidity.csv
│    └─── pressure.csv
│    └─── temperature.csv
│    └─── weather_description.csv
│    └─── wind_direction.csv
│    └─── wind_speed.csv
│
└─── ...
"""


def download_file_from_google_drive(id, destination):
    
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):

    CHUNK_SIZE = 32768
    # Total size of the downlaoded zip file:
    total_size = 12556386    
    
    pbar = tqdm(total=total_size, unit='iB', unit_scale=True)    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            # Filter out keep-alive new chunks:
            if chunk: 
                f.write(chunk)
                pbar.set_description("Downloading dataset...")
                pbar.update(CHUNK_SIZE)
        pbar.set_description_str("Download completed...")


def download_dataset(dataset_path, destination):
    
    file_id = '1CvgryzWKooAtdTtaQ4Rumo9988TePa8-'
    download_file_from_google_drive(file_id, destination)
    
    with ZipFile(destination, 'r') as zipObj:
        # Extract all the contents of zip file in current directory:   
        zipObj.extractall(dataset_path)
        print("Extraction completed...")


def check_dataset(dataset_path, zip_path):
    """
    Checks if dataset exists. If not downloads it.
    """
    necessary_files = ['city_attributes.csv', 'humidity.csv', 'pressure.csv', 
                       'temperature.csv', 'weather_description.csv', 
                       'wind_direction.csv', 'wind_speed.csv']

    for file in necessary_files:
        file_check = os.path.isfile(os.path.join(dataset_path, file))
        
        if file_check == False:
            download_dataset(dataset_path=dataset_path, destination=zip_path)            
            return check_dataset(dataset_path=dataset_path, zip_path=zip_path)
    
    print("Dataset ready...")
    return True


def main():

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    check_dataset(dataset_path=DATASET_DIR, zip_path=ZIP_PATH)


if __name__ == "__main__":
    main()














