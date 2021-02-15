# 2:  humidity               (min: 5,        max: 100)
# 3:  pressure               (min: 800,      max: 1100)
# 4:  temperature            (min: 242.3367, max: 321.22)
# 5:  weather description    (min: 0,        max: 53)
# 6:  wind direction         (min: 0,        max: 360)
# 7:  wind speed             (min: 0,        max: 50)
features_min_max_values = {
    2: (5, 100),
    3: (800, 1100),
    4: (242.3367, 321.22),
    5: (0, 53),
    6: (0, 360),
    7: (0, 50),
}


def denormalize_feature(values, feature_index=None, min=None, max=None):
    assert feature_index is not None or (min is not None and max is not None)

    if feature_index is not None:
        if feature_index not in features_min_max_values:
            print(f'Feature not suitable for denormalization.')
            return values
        value_min, value_max = features_min_max_values[feature_index]

    else:
        value_min, value_max = min, max

    return values * (value_max - value_min) + value_min
