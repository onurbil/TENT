city_labels = ['Vancouver', 'Portland', 'San Francisco', 'Seattle', 'Los Angeles',
'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque', 'Denver', 'San Antonio',
'Dallas', 'Houston', 'Kansas City', 'Minneapolis', 'Saint Louis', 'Chicago',
'Nashville', 'Indianapolis', 'Atlanta', 'Detroit', 'Jacksonville', 'Charlotte',
'Miami', 'Pittsburgh', 'Toronto', 'Philadelphia', 'New York', 'Montreal',
'Boston', 'Beersheba', 'Tel Aviv District', 'Eilat', 'Haifa', 'Nahariyya',
'Jerusalem']

eu_city_labels = ['Amsterdam', 'Barcelona', 'Berlin', 'Brussels', 'Copenhagen',
'Dublin', 'Frankfurt', 'Hamburg', 'London', 'Luxembourg', 'Lyon', 'Maastricht',
'Malaga', 'Marseille', 'Munich', 'Nice', 'Paris', 'Rotterdam']

weather_desc_labels = {'shower drizzle':0, 'freezing rain':1, 'volcanic ash':2,
'proximity shower rain':3, 'fog':4, 'shower snow':5, 'tornado':6, 'drizzle':7,
'heavy shower snow':8, 'few clouds':9, 'proximity sand/dust whirls':10,
'mist':11, 'light rain':12, 'light shower sleet':13, 'rain and snow':14,
'proximity thunderstorm with rain':15, 'thunderstorm with heavy drizzle':16,
'overcast clouds':17, 'sky is clear':18, 'light rain and snow':19,
'proximity moderate rain':20, 'light intensity drizzle rain':21,
'heavy thunderstorm':22, 'thunderstorm with rain':23, 'scattered clouds':24,
'sand/dust whirls':25, 'moderate rain':26, 'broken clouds':27, 'shower rain':28,
'smoke':29, 'haze':30, 'heavy intensity shower rain':31, 'sleet':32,
'squalls':33, 'heavy snow':34, 'sand':35, 'ragged shower rain':36,
'thunderstorm with heavy rain':37, 'ragged thunderstorm':38,
'thunderstorm with light rain':39, 'thunderstorm with light drizzle':40,
'light intensity shower rain':41, 'snow':42, 'heavy intensity rain':43,
'light shower snow':44, 'thunderstorm with drizzle':45,
'heavy intensity drizzle':46, 'thunderstorm':47, 'light snow':48,
'proximity thunderstorm':49, 'light intensity drizzle':50, 'dust':51,
'proximity thunderstorm with drizzle':52, 'very heavy rain':53}


# DATASET_SAMPLE_SIZE is the row size in the dataset (for geolocation.py):
DATASET_SAMPLE_SIZE = 45253
# Sum of train and validation samples:
TRAIN_VAL_SIZE = 36333

# DATASET_EU:SAMPLE_SIZE is the row size in the dataset (for eu_geolocation.py):
EU_DATASET_SAMPLE_SIZE = 5470
# Sum of train and validation samples:
EU_TRAIN_VAL_SIZE = 4375
