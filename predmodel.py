import math
import os

import numpy as np
from sklearn.externals import joblib


class PredModel():

    model = None

    def __init__(self, model_file):
        model_file = os.path.join(os.getcwd(), 'models', model_file)
        self.model = joblib.load(model_file)
        print('Model loaded')

    def predict(self, w):
        w_orig = w.copy()
        fixed = self.feat_eng(w)
        prediction = self.model.predict([fixed])[0]
        # TODO update to take an array of weathers
        w_orig['relative_humidity_pm_prediction'] = prediction
        return w_orig

    @staticmethod
    def _rotate_to_x_axis(angle):
        """Rotate such that East is 0 and North is 90
        Shouldn't affect the outcome but makes it more human understandable.
        """
        return (90.0 + 360.0 - angle) % 360

    @staticmethod
    def _get_x_component(direction, magnitude):
        x_angle = PredModel._rotate_to_x_axis(direction)
        return magnitude * math.cos(x_angle)

    @staticmethod
    def _get_y_component(direction, magnitude):
        x_angle = PredModel._rotate_to_x_axis(direction)
        return magnitude * math.sin(x_angle)

    @staticmethod
    def _water_vapour_pressure(t):
        """ Get WVP (in mmHg)
        t - temp in .C
        Calc water vapour pressure
        wvp = 610.17 * e^((17.2694 * t)/(t + 238.3))
        """
        if t is None:
            return None
        ex_pwr = (17.2694 * t)/(t + 238.3)
        return 610.17 * np.exp(ex_pwr)

    _mean_map = {
        'air_pressure': 918.883467,
        'air_temperature': 18.296112,
        'rain_accumulation': 0.213538,
        'rain_duration': 294.111715,
        'relative_humidity': 34.242315,
        'relative_humidity_pm': 35.346554,
        'avg_wind_x': -0.411050,
        'avg_wind_y': -0.065121,
        'max_wind_x': -0.167812,
        'max_wind_y': 0.180229,
        'avg_direction_north_south': 0.239709,
        'avg_direction_east_west': -0.354527,
        'max_direction_north_south': 0.201641,
        'max_direction_east_west': -0.378705,
        'has_rain': 0.168950,
        'rain_accumulation_small': 0.072146,
        'rain_accumulation_big': 0.050228,
        'rain_duration_small': 0.087671,
        'rain_duration_big': 0.081279,
        'water_vapour_pressure': 2227.068041,
        'actual_water_vapour_pressure_transform': 6.314475,
        'avg_wind_speed_transform': 1.410237,
        'max_wind_speed_transform': 1.684621
    }

    _stdev_map = {
        'air_pressure': 3.184068,
        'air_temperature': 6.208619,
        'rain_accumulation': 1.597062,
        'rain_duration': 1598.078108,
        'relative_humidity': 25.472378,
        'relative_humidity_pm': 22.523507,
        'avg_wind_x': 4.998785,
        'avg_wind_y': 5.088263,
        'max_wind_x': 6.150300,
        'max_wind_y': 6.542323,
        'avg_direction_north_south': 0.581798,
        'avg_direction_east_west': 0.692178,
        'max_direction_north_south': 0.680224,
        'max_direction_east_west': 0.594951,
        'has_rain': 0.374879,
        'rain_accumulation_small': 0.258848,
        'rain_accumulation_big': 0.218516,
        'rain_duration_small': 0.282945,
        'rain_duration_big': 0.273387,
        'water_vapour_pressure': 830.528217,
        'actual_water_vapour_pressure_transform': 0.595272,
        'avg_wind_speed_transform': 0.763797,
        'max_wind_speed_transform': 0.710755
    }

    @staticmethod
    def _fix_nulls(w):
        for field in w:
            if w[field] is None:
                w[field] = PredModel._mean_map[field]
        return w

    @staticmethod
    def _normalize(w):
        for field in w:
            mean = PredModel._mean_map[field]
            std = PredModel._stdev_map[field]
            w[field] = (w[field] - mean) / std

        return w

    def feat_eng(self, w):
        del w['feed_timestamp']
        del w['last_hash']
        del w['row_number']

        w = self._fix_nulls(w)

        w['avg_wind_x'] = self._get_x_component(
            w['avg_wind_direction'], w['avg_wind_speed'])
        w['avg_wind_y'] = self._get_y_component(
            w['avg_wind_direction'], w['avg_wind_speed'])
        w['max_wind_x'] = self._get_x_component(
            w['max_wind_direction'], w['max_wind_speed'])
        w['max_wind_y'] = self._get_y_component(
            w['max_wind_direction'], w['max_wind_speed'])

        w['avg_direction_rads'] = np.deg2rad(w['avg_wind_direction'])

        w['avg_direction_north_south'] = np.sin(w['avg_direction_rads'])
        w['avg_direction_east_west'] = np.cos(w['avg_direction_rads'])

        w['max_direction_north_south'] = np.sin(np.deg2rad(
            w['max_wind_direction']))
        w['max_direction_east_west'] = np.cos(np.deg2rad(
            w['max_wind_direction']))

        del w['avg_direction_rads']

        # Drop the missleading direction columns
        del w['avg_wind_direction']
        del w['max_wind_direction']

        w['has_rain'] = 1 if w['rain_duration'] > 0.0 else 0

        # Then split the other rain columns by the non zero values
        ra = w['rain_accumulation']
        w['rain_accumulation_small'] = 1 if (ra <= 0.3 and ra > 0) else 0
        w['rain_accumulation_big'] = 1 if ra > 0.3 else 0

        rd = w['rain_duration']
        w['rain_duration_small'] = 1 if (rd <= 150 and rd > 0) else 0
        w['rain_duration_big'] = 1 if rd > 150 else 0

        w['water_vapour_pressure'] = self._water_vapour_pressure(
            w['air_temperature'])

        w['actual_water_vapour_pressure'] = \
            w['relative_humidity']/100 * w['water_vapour_pressure']

        w['actual_water_vapour_pressure_transform'] = \
            np.log(w['actual_water_vapour_pressure'])
        w['avg_wind_speed_transform'] = np.log(w['avg_wind_speed'])
        w['max_wind_speed_transform'] = np.log(w['max_wind_speed'])

        del w['actual_water_vapour_pressure']
        del w['avg_wind_speed']
        del w['max_wind_speed']

        w = self._normalize(w)

        return self._to_array(w)

    def _to_array(self, w):
        cols = [
            'max_direction_north_south', 'rain_duration_small',
            'actual_water_vapour_pressure_transform',
            'max_direction_east_west',
            'rain_duration_big', 'avg_direction_north_south', 'air_pressure',
            'avg_wind_y', 'avg_direction_east_west', 'rain_duration',
            'has_rain', 'rain_accumulation_small', 'water_vapour_pressure',
            'air_temperature', 'avg_wind_x', 'max_wind_speed_transform',
            'rain_accumulation', 'avg_wind_speed_transform',
            'rain_accumulation_big', 'max_wind_y', 'relative_humidity',
            'max_wind_x'
        ]
        output = []
        for col in cols:
            output.append(w[col])
        return output


if __name__ == "__main__":
    model_file = 'forest_reg.joblib'
    pm = PredModel(model_file)

    msgs = [
        b'{"air_pressure": 919.06, "air_temperature": 23.79, "avg_wind_direction": 281.1, "avg_wind_speed": 3.0803542, "max_wind_direction": 275.4, "max_wind_speed": 3.8632832, "rain_accumulation": 0.89, "rain_duration": 4.0, "relative_humidity": 43.42, "row_number": 0.0, "feed_timestamp": 1559859360.0775285, "last_hash": 0}\n',
        b'{"air_pressure": 917.3476881, "air_temperature": 21.89102368, "avg_wind_direction": 101.9351794, "avg_wind_speed": 2.443009216, "max_wind_direction": 140.4715485, "max_wind_speed": 3.533323602, "rain_accumulation": 0.0, "rain_duration": 0.0, "relative_humidity": 24.32869729, "row_number": 1.0, "feed_timestamp": 1559859361.0787408, "last_hash": 5010695143364537226}\n',
        b'{"air_pressure": 923.04, "air_temperature": 15.91, "avg_wind_direction": 51.0, "avg_wind_speed": 17.0678522, "max_wind_direction": 63.7, "max_wind_speed": 22.1009672, "rain_accumulation": 0.0, "rain_duration": 20.0, "relative_humidity": 8.9, "row_number": 2.0, "feed_timestamp": 1559859362.0799596, "last_hash": 3377101931316371610}\n',
        b'{"air_pressure": 920.5027512, "air_temperature": 21.18827493, "avg_wind_direction": 198.8321327, "avg_wind_speed": 4.337363056, "max_wind_direction": 211.2033412, "max_wind_speed": 5.19004536, "rain_accumulation": 0.0, "rain_duration": 0.0, "relative_humidity": 12.18910187, "row_number": 3.0, "feed_timestamp": 1559859363.081159, "last_hash": 5637290165899419028}\n',
        b'{"air_pressure": 921.16, "air_temperature": 6.83, "avg_wind_direction": 277.8, "avg_wind_speed": 1.8566602, "max_wind_direction": 136.5, "max_wind_speed": 2.8632832, "rain_accumulation": 8.9, "rain_duration": 14730.0, "relative_humidity": 92.41, "row_number": 4.0, "feed_timestamp": 1559859364.082418, "last_hash": -822556792205670904}\n',
        b'{"air_pressure": 915.3, "air_temperature": 25.78, "avg_wind_direction": 182.8, "avg_wind_speed": 9.9320136, "max_wind_direction": 189.0, "max_wind_speed": 10.9833754, "rain_accumulation": 0.02, "rain_duration": 170.0, "relative_humidity": 35.13, "row_number": 5.0, "feed_timestamp": 1559859365.0839045, "last_hash": -3913013259966945264}\n'
    ]

    import json
    for msg in msgs:
        w = json.loads(msg)
        prediction = pm.predict(w)
        print(prediction)
