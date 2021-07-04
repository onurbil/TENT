from dataset_tools import dataset, encode_weather, fill_nan, to_tensor, geolocation, eu_download, eu_dataset
from model import tensorized_transformer
# 1. USA+Canada Dataset
dataset.main()
encode_weather.main()
fill_nan.main()
geolocation.main()
to_tensor.main()

# 2. Europe Dataset
eu_download.main()
eu_dataset.main()


# TENT example training for USA+Canada dataset
tensorized_transformer.main()