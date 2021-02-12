from dataset_tools import dataset, encode_weather, fill_nan, to_tensor, geolocation, eu_download, eu_dataset

# 1. Dataset
dataset.main()
encode_weather.main()
fill_nan.main()
geolocation.main()
to_tensor.main()
# 2. Dataset
eu_download.main()
eu_dataset.main()
