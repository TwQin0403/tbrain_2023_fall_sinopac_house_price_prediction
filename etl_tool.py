import pandas as pd
import numpy as np
import pyproj
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from collections import UserDict
import dataloader
import bisect


def get_transfer_year(transfer_date):
    if isinstance(transfer_date, int):
        return np.nan
    else:
        return transfer_date.split('-')[0]


class PostStepTransformer():

    def __init__(self):
        loader = dataloader.DataLoader()
        train = loader.load_train_data('training_data.csv', data_type='csv')
        train = preprocess_raw_data(train, is_simple=True)
        self.possible_values = list(train['y'].sort_values().unique())

    def find_closest(self, target):
        idx = bisect.bisect_left(self.possible_values, target)
        if idx == 0:
            return self.possible_values[0]
        elif idx == len(self.possible_values):
            return self.possible_values[-1]
        else:
            before = self.possible_values[idx - 1]
            after = self.possible_values[idx]
            return before if target - before <= after - target else after

    def transform(self, preds):
        return [self.find_closest(pred) for pred in preds]


def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    R = 6371

    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)

    a = (np.sin(d_lat / 2)**2 + np.cos(np.radians(lat1)) *
         np.cos(np.radians(lat2)) * np.sin(d_lon / 2)**2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance


def get_preprocessing_transformer():
    transformer = Pipeline([
        ('other_same_budiling_price', OtherSameBuildingRecord()),
        ('same_record_price', OtherPossibleSameRecord()),
        ('otehr_price_same', OtherSameHouseTransformer()),
        ('other_price_similar', OtherSimilarHouseTransformer()),
        ('other_price_mean', OtherHouseTransformerNew()),
        ('other_price_mean_all', OtherHouseTransformer()),
        ('recreation_center_transformer', RecreationCenterTransformer()),
        ("CommunityTransformer", CommunityTransformer()),
        ("senior_high_transformer", SeniorHighSchoolTransformer()),
        ("university_transformer", UniversityTransformer()),
        ("high_school_transformer", HighSchoolTransformer()),
        ("hospital_transformer", HospitalTransformer()),
        ("train_transformer", TrainTransformer()),
        ("MRT_transformer", MRTAllTransformer()),
        ('other_similar_building_price', OtherSimilarBuildingTransformer()),
    ])
    return transformer


def preprocess_raw_data(df: pd.DataFrame, is_simple=False) -> pd.DataFrame:
    use_df = df.copy()
    use_df.rename(columns={
        "ID": "id",
        "縣市": "city",
        "鄉鎮市區": "district",
        "路名": "road",
        "土地面積": "land_area",
        "使用分區": "usage_zone",
        "移轉層次": "transfer_floor_level",
        "總樓層數": "total_floors",
        "主要用途": "primary_use",
        "主要建材": "main_material",
        "建物型態": "building_type",
        "屋齡": "building_age",
        "建物面積": "building_area",
        "車位面積": "parking_space_area",
        "車位個數": "number_of_parking_spaces",
        "橫坐標": "x_coordinate",
        "縱坐標": "y_coordinate",
        "備註": "notes",
        "主建物面積": "main_building_area",
        "陽台面積": "balcony_area",
        "附屬建物面積": "attached_building_area",
        "單價": "y"
    },
                  inplace=True)
    use_df['area'] = use_df['city'] + use_df['district']
    use_df['area_road'] = use_df['city'] + use_df['district'] + use_df['road']
    wgs84 = pyproj.CRS("EPSG:4326")
    twd97 = pyproj.CRS("EPSG:3826")
    cord_transformer = pyproj.Transformer.from_crs(twd97,
                                                   wgs84,
                                                   always_xy=True)
    lons = []
    lats = []
    for x, y in zip(use_df['x_coordinate'], use_df['y_coordinate']):
        lon, lat = cord_transformer.transform(x, y)
        lons.append(lon)
        lats.append(lat)
    use_df['lon'] = lons
    use_df['lat'] = lats
    use_df.loc[:, 'is_notes'] = use_df['notes'].notna()
    use_df.loc[:, 'is_notes'] = use_df['is_notes'].astype(int)
    use_df['transfer_level_ratio'] = use_df['transfer_floor_level'] / use_df[
        'total_floors']
    if not is_simple:
        transformer = get_preprocessing_transformer()
        use_df = transformer.fit_transform(use_df)
    return use_df


class SameHouseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.output_name = 'y_same_house_mean'

    def fit(self, X, y=None, **fit_params):
        X_use = X.copy()
        y_use = y.copy()
        self.df = pd.concat([X_use, y_use], axis=1)
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        df = self.df.copy()
        prices = []
        for a_id, area_road, total_floors, building_age, primary_use, main_material, building_type, transfer_floor_level in zip(  # noqa
                X_copy['id'], X_copy['area_road'], X_copy['total_floors'],
                X_copy['building_age'], X_copy['primary_use'],
                X_copy['main_material'], X_copy['building_type'],
                X_copy['transfer_floor_level']):  # noqa
            a_df = df[(df['area_road'] == area_road)
                      & (df['total_floors'] == total_floors) &
                      (df['primary_use'] == primary_use) &
                      (df['building_type'] == building_type) &
                      (df['main_material'] == main_material) &
                      (df['transfer_floor_level']
                       == transfer_floor_level)].copy()
            a_df = a_df[(a_df['building_age'] <= building_age + 1)
                        & (a_df['building_age'] >= building_age - 1)]
            a_df = a_df[a_df['id'] != a_id]
            if len(a_df) == 0:
                prices.append(np.nan)
            else:
                prices.append(a_df['y'].mean())
        X_copy[self.output_name] = prices
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class SimilarHouseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.output_name = 'y_similar_house_mean'

    def fit(self, X, y=None, **fit_params):
        X_use = X.copy()
        y_use = y.copy()
        self.df = pd.concat([X_use, y_use], axis=1)
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        df = self.df.copy()
        prices = []
        for a_id, area_road, building_age, primary_use, main_material, building_type in zip(  # noqa
                X_copy['id'],
                X_copy['area_road'],
                X_copy['building_age'],
                X_copy['primary_use'],
                X_copy['main_material'],
                X_copy['building_type'],
        ):  # noqa
            a_df = df[(df['area_road'] == area_road)
                      & (df['primary_use'] == primary_use) &
                      (df['building_type'] == building_type) &
                      (df['main_material'] == main_material)].copy()
            a_df = a_df[(a_df['building_age'] <= building_age + 0.5)
                        & (a_df['building_age'] >= building_age - 0.5)]
            a_df = a_df[a_df['id'] != a_id]
            if len(a_df) == 0:
                prices.append(np.nan)
            else:
                prices.append(a_df['y'].mean())
        X_copy[self.output_name] = prices
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class PostSameHouseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        train = loader.load_train_data('training_data.csv', data_type='csv')
        test = loader.load_train_data('public_dataset.csv', data_type='csv')
        private = loader.load_train_data('private_dataset.csv',
                                         data_type='csv')

        self.train = preprocess_raw_data(train, is_simple=True)
        self.private = preprocess_raw_data(private, is_simple=True)
        self.test = preprocess_raw_data(test, is_simple=True)
        self.test = pd.concat([self.test, self.private]).reset_index(drop=True)
        self.test.loc[:, 'y'] = -1
        self.df = pd.concat([self.train, self.test]).reset_index(drop=True)

    def fit(self, X, y=None, **fit_params):
        df = self.df.copy()
        group = df.groupby(['area_road', 'building_age', 'total_floors'])
        use_mapping = {}
        for a_df in group:
            if (len(a_df[1]) > 1) and (a_df[1]['y'].min() == -1):
                if len(a_df[1][a_df[1]['y'] != -1]) > 0:
                    if a_df[1]['main_building_area'].std() <= 0.55:
                        use_mean = a_df[1][a_df[1]['y'] != -1]['y'].mean()
                        for a_id in a_df[1][a_df[1]['y'] == -1]['id']:
                            use_mapping.update({a_id: use_mean})
        self.use_mapping = use_mapping
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        for a_id, pred in zip(X_copy['id'], X_copy['y']):
            if a_id in self.use_mapping.keys():
                X_copy.loc[X_copy['id'] == a_id, 'y'] = self.use_mapping[a_id]
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class RecreationCenterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        bus_station = loader.load_train_data('external_data/公車站點資料.csv',
                                             data_type='csv')

        community = bus_station[(bus_station['站點名稱'].str.contains('運動中心')) |
                                (bus_station['站點名稱'].str.contains('活動中心'))]
        self.community = community.groupby('站點名稱').head(1).reset_index(
            drop=True)
        self.output_1 = "recreation_center_dist"
        self.output_2 = "recreation_center_name"

    def compute_a_single(self, lat1, lon1):
        parks = self.community.copy()
        dists = haversine_distance_vectorized(lat1, lon1, parks['lat'],
                                              parks['lng'])
        parks['dist'] = dists
        min_dist_idx = parks['dist'].idxmin()
        nearest_park = parks.loc[min_dist_idx, '站點名稱']
        return nearest_park, parks.loc[min_dist_idx, 'dist']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        schools = []
        dists = []
        for lat, lng in zip(X_copy['lat'], X_copy['lon']):
            school, dist = self.compute_a_single(lat, lng)
            schools.append(school)
            dists.append(dist)
        X_copy[self.output_1] = dists
        X_copy[self.output_2] = schools
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class CommunityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        bus_station = loader.load_train_data('external_data/公車站點資料.csv',
                                             data_type='csv')

        # parks = bus_station[bus_station['站點名稱'].str.contains('公園')]

        community = bus_station[bus_station['站點名稱'].str.contains('社區')]
        self.community = community.groupby('站點名稱').head(1).reset_index(
            drop=True)
        self.output_1 = "community_dist"
        self.output_2 = "community_name"

    def compute_a_single(self, lat1, lon1):
        parks = self.community.copy()
        dists = haversine_distance_vectorized(lat1, lon1, parks['lat'],
                                              parks['lng'])
        parks['dist'] = dists
        min_dist_idx = parks['dist'].idxmin()
        nearest_park = parks.loc[min_dist_idx, '站點名稱']
        return nearest_park, parks.loc[min_dist_idx, 'dist']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        schools = []
        dists = []
        for lat, lng in zip(X_copy['lat'], X_copy['lon']):
            school, dist = self.compute_a_single(lat, lng)
            schools.append(school)
            dists.append(dist)
        X_copy[self.output_1] = dists
        X_copy[self.output_2] = schools
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class PossibeSameBuilding(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.output_name = 'y_same_building_mean'

    def fit(self, X, y=None, **fit_params):
        X_use = X.copy()
        y_use = y.copy()
        self.df = pd.concat([X_use, y_use], axis=1)
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        df = self.df.copy()
        prices = []
        for a_id, area_road, total_floors, land_area, building_age in zip(  # noqa
                X_copy['id'], X_copy['area_road'], X_copy['total_floors'],
                X_copy['land_area'], X_copy['building_age']):  # noqa
            a_df = df[(df['area_road'] == area_road)
                      & (df['total_floors'] == total_floors) &
                      (df['land_area'] == land_area)].copy()
            a_df = a_df[(a_df['building_age'] <= building_age + 0.5)
                        & (a_df['building_age'] >= building_age - 0.5)]
            a_df = a_df[a_df['id'] != a_id]
            if len(a_df) == 0:
                prices.append(np.nan)
            else:
                prices.append(a_df['y'].mean())
        X_copy[self.output_name] = prices
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class OtherSameBuildingRecord(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.df = loader.load_input('other_prices.joblib')
        self._preprocess_data()
        self.df['year'] = self.df['transfer_date'].apply(get_transfer_year)
        self.df = self.df[self.df['year'].isin(['2021', '2022'])]

    def _preprocess_data(self):
        self.df['remark'] = self.df['remark'].fillna('no_info')
        self.df = self.df[~self.df['remark'].str.contains('特殊')]
        self.df = self.df[~self.df['remark'].str.contains('攤位')]
        self.df = self.df[~self.df['remark'].str.contains('親友')]
        self.df = self.df[~self.df['remark'].str.contains('瑕疵')]
        self.df = self.df[~self.df['remark'].str.contains('身故')]
        self.df['price'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna(subset=['price'])
        self.df['price'] = self.df['price'].astype(int)
        self.df = self.df[self.df['price'] > 0]
        self.df['established_date'] = self.df['established_date'].apply(
            self.roc_to_ad)
        self.df['transfer_date'] = self.df['transfer_date'].apply(
            self.roc_to_ad)
        self.df['building_age'] = self.df.apply(lambda x: self.compute_age(
            x['transfer_date'], x['established_date']),
                                                axis=1)
        self.df['building_area'] = self.df['building_area'].clip(upper=1300)
        self.df['building_area'] = (self.df['building_area'] / 100) - 2
        self.df['building_area_rank'] = self.df['building_area'].rank(pct=True)

    def roc_to_ad(self, roc_date):
        if not isinstance(roc_date, str):
            return -1
        if roc_date == 'nan':
            return -1
        if roc_date[0] == "0":  # format: 0yyMMdd
            roc_year = int(roc_date[1:3])
        else:  # format: yyyMMdd
            try:
                roc_year = int(roc_date[:3])
            except ValueError:
                return -1
        if '.' in roc_date:
            roc_date = roc_date.split('.')[0]

        ad_year = 1911 + roc_year
        if ad_year > 2023:
            return -1
        ad_date = str(ad_year) + '-' + '01' + '-' + '01'
        return ad_date

    def compute_age(self, year, established_date):
        if established_date == -1:
            return -1
        year = pd.to_datetime(year)
        established_date = pd.to_datetime(established_date)
        days = (year - established_date).days
        days = days / 365
        return days

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        result = {}

        for year in self.df['year'].unique():
            df_filtered = self.df.loc[self.df['year'] == year].copy()
            df_filtered = df_filtered[df_filtered['building_age'] != -1]
            n_year = int(year)
            delta_year_min = 2021 - n_year
            delta_year_max = 2022 - n_year

            prices = []
            for _, row in X_copy.iterrows():
                mask = ((df_filtered['transfer_floor_level']
                         == row['transfer_floor_level'])
                        & (df_filtered['area_road'] == row['area_road']) &
                        (df_filtered['total_floors'] == row['total_floors']) &
                        (df_filtered['building_age'].between(
                            row['building_age'] - delta_year_max - 0.5,
                            row['building_age'] - delta_year_min + 0.5)))
                subset = df_filtered.loc[mask, 'price']
                prices.append(subset.median() if not subset.empty else np.nan)
            result[f"other_same_building_{year}_mean"] = prices
        X_copy = X_copy.assign(**result)
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class OtherPossibleSameRecord(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.df = loader.load_input('other_prices.joblib')
        self.df['remark'] = self.df['remark'].fillna('no_info')
        self.df = self.df[~self.df['remark'].str.contains('特殊')]
        self.df = self.df[~self.df['remark'].str.contains('攤位')]
        self.df = self.df[~self.df['remark'].str.contains('親友')]
        self.df = self.df[~self.df['remark'].str.contains('瑕疵')]
        self._preprocess_data()
        self.df['year'] = self.df['transfer_date'].apply(get_transfer_year)
        self.df = self.df[self.df['year'].isin(
            ['2020', '2021', '2022', '2023'])]

    def roc_to_ad(self, roc_date):
        if not isinstance(roc_date, str):
            return -1
        if roc_date == 'nan':
            return -1
        if roc_date[0] == "0":  # format: 0yyMMdd
            roc_year = int(roc_date[1:3])
        else:  # format: yyyMMdd
            try:
                roc_year = int(roc_date[:3])
            except ValueError:
                return -1
        if '.' in roc_date:
            roc_date = roc_date.split('.')[0]

        ad_year = 1911 + roc_year
        if ad_year > 2023:
            return -1
        ad_date = str(ad_year) + '-' + '01' + '-' + '01'
        return ad_date

    def compute_age(self, year, established_date):
        if established_date == -1:
            return -1
        year = pd.to_datetime(year)
        established_date = pd.to_datetime(established_date)
        days = (year - established_date).days
        days = days / 365
        return days

    def _preprocess_data(self):
        self.df['remark'] = self.df['remark'].fillna('no_info')
        self.df = self.df[~self.df['remark'].str.contains('特殊')]
        self.df = self.df[~self.df['remark'].str.contains('攤位')]
        self.df = self.df[~self.df['remark'].str.contains('親友')]
        self.df = self.df[~self.df['remark'].str.contains('瑕疵')]
        self.df = self.df[~self.df['remark'].str.contains('身故')]
        self.df = self.df[~self.df['remark'].str.contains('預售屋')]
        self.df['price'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna(subset=['price'])
        self.df['price'] = self.df['price'].astype(int)
        self.df = self.df[self.df['price'] > 0]
        self.df['established_date'] = self.df['established_date'].apply(
            self.roc_to_ad)
        self.df['transfer_date'] = self.df['transfer_date'].apply(
            self.roc_to_ad)
        self.df['building_age'] = self.df.apply(lambda x: self.compute_age(
            x['transfer_date'], x['established_date']),
                                                axis=1)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        merge_copy = X[[
            'area_road', 'total_floors', 'transfer_floor_level',
            'building_age', 'building_type'
        ]].copy()
        df = self.df.copy()
        df['building_age'] = df['building_age'].astype(int)
        merge_copy['building_age'] = merge_copy['building_age'].astype(int)

        results = pd.merge(df,
                           merge_copy,
                           on=[
                               'area_road', 'total_floors',
                               'transfer_floor_level', 'building_type',
                               'building_age'
                           ],
                           how='inner')
        price_map = results.groupby([
            'area_road', 'total_floors', 'transfer_floor_level',
            'building_type'
        ])
        price_map = {df[0]: df[1]['price'].median() for df in price_map}
        X_copy['other_possible_same_record'] = X_copy.set_index([
            'area_road', 'total_floors', 'transfer_floor_level',
            'building_type'
        ]).index.map(price_map).values
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class OtherSameHouseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.df = loader.load_input('other_prices.joblib')
        self.df = self.df[self.df['transfer_floor_level'] != 1]
        self.df = self.df[self.df['building_type'].isin(
            ['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)', '公寓(5樓含以下無電梯)', '透天厝'])]
        self._preprocess_data()
        self.df['year'] = self.df['transfer_date'].apply(get_transfer_year)
        self.df = self.df[self.df['year'].isin(
            ['2021', '2022', '2023', '2019', '2020'])]

    def _preprocess_data(self):
        self.df['remark'] = self.df['remark'].fillna('no_info')
        self.df = self.df[~self.df['remark'].str.contains('特殊')]
        self.df = self.df[~self.df['remark'].str.contains('攤位')]
        self.df = self.df[~self.df['remark'].str.contains('親友')]
        self.df = self.df[~self.df['remark'].str.contains('瑕疵')]
        self.df = self.df[~self.df['remark'].str.contains('身故')]
        # self.df = self.df[~self.df['transfer_floor_level'] != 1]
        self.df['price'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna(subset=['price'])
        self.df['price'] = self.df['price'].astype(int)
        self.df = self.df[self.df['price'] > 0]
        self.df['established_date'] = self.df['established_date'].apply(
            self.roc_to_ad)
        self.df['transfer_date'] = self.df['transfer_date'].apply(
            self.roc_to_ad)
        self.df['building_age'] = self.df.apply(lambda x: self.compute_age(
            x['transfer_date'], x['established_date']),
                                                axis=1)
        self.df['building_area'] = self.df['building_area'].clip(upper=1300)
        self.df['building_area'] = (self.df['building_area'] / 100) - 2
        self.df['building_area_rank'] = self.df['building_area'].rank(pct=True)

    def roc_to_ad(self, roc_date):
        if not isinstance(roc_date, str):
            return -1
        if roc_date == 'nan':
            return -1
        if roc_date[0] == "0":  # format: 0yyMMdd
            roc_year = int(roc_date[1:3])
        else:  # format: yyyMMdd
            try:
                roc_year = int(roc_date[:3])
            except ValueError:
                return -1
        if '.' in roc_date:
            roc_date = roc_date.split('.')[0]

        ad_year = 1911 + roc_year
        if ad_year > 2023:
            return -1
        ad_date = str(ad_year) + '-' + '01' + '-' + '01'
        return ad_date

    def compute_age(self, year, established_date):
        if established_date == -1:
            return -1
        year = pd.to_datetime(year)
        established_date = pd.to_datetime(established_date)
        days = (year - established_date).days
        days = days / 365
        return days

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        result = {}

        for year in self.df['year'].unique():
            df_filtered = self.df.loc[self.df['year'] == year].copy()
            df_filtered = df_filtered[df_filtered['building_age'] != -1]
            n_year = int(year)
            delta_year_min = 2021 - n_year
            delta_year_max = 2022 - n_year

            prices = []
            for _, row in X_copy.iterrows():
                mask = (df_filtered['area_road'] == row['area_road']) & (
                    df_filtered['total_floors'] == row['total_floors']) & (
                        df_filtered['building_age'].between(
                            row['building_age'] - delta_year_max - 0.5,
                            row['building_age'] - delta_year_min + 0.5))
                subset = df_filtered.loc[mask, 'price']
                prices.append(subset.median() if not subset.empty else np.nan)
            result[f"other_same_house_{year}_mean"] = prices
        X_copy = X_copy.assign(**result)
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class OtherSimilarBuildingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.df = loader.load_input('other_prices.joblib')
        self._preprocess_data()
        self.df = self.df[self.df['year'].isin(
            ['2020', '2021', '2022', '2023'])]

    def _preprocess_data(self):
        self.df['remark'] = self.df['remark'].fillna('no_info')
        self.df = self.df[~self.df['remark'].str.contains('特殊')]
        self.df = self.df[~self.df['remark'].str.contains('攤位')]
        self.df = self.df[~self.df['remark'].str.contains('親友')]
        self.df = self.df[~self.df['remark'].str.contains('瑕疵')]
        self.df = self.df[~self.df['remark'].str.contains('身故')]
        self.df['price'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna(subset=['price'])
        self.df['price'] = self.df['price'].astype(int)
        self.df = self.df[self.df['price'] > 0]
        self.df['established_date'] = self.df['established_date'].apply(
            self.roc_to_ad)
        self.df['transfer_date'] = self.df['transfer_date'].apply(
            self.roc_to_ad)
        self.df['building_age'] = self.df.apply(lambda x: self.compute_age(
            x['transfer_date'], x['established_date']),
                                                axis=1)
        self.df['building_area'] = self.df['building_area'].clip(upper=1300)
        self.df['building_area'] = (self.df['building_area'] / 100) - 2
        self.df['building_area_rank'] = self.df['building_area'].rank(pct=True)

    def roc_to_ad(self, roc_date):
        if not isinstance(roc_date, str):
            return -1
        if roc_date == 'nan':
            return -1
        if roc_date[0] == "0":  # format: 0yyMMdd
            roc_year = int(roc_date[1:3])
        else:  # format: yyyMMdd
            try:
                roc_year = int(roc_date[:3])
            except ValueError:
                return -1
        if '.' in roc_date:
            roc_date = roc_date.split('.')[0]

        ad_year = 1911 + roc_year
        if ad_year > 2023:
            return -1
        ad_date = str(ad_year) + '-' + '01' + '-' + '01'
        return ad_date

    def compute_age(self, year, established_date):
        if established_date == -1:
            return -1
        year = pd.to_datetime(year)
        established_date = pd.to_datetime(established_date)
        days = (year - established_date).days
        days = days / 365
        return days

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        result = {}

        for year in self.df['year'].unique():
            df_filtered = self.df.loc[self.df['year'] == year].copy()
            df_filtered = df_filtered[df_filtered['building_age'] != -1]
            prices = []
            # stds = []
            for _, row in X_copy.iterrows():

                mask = (
                    (df_filtered['area_road'] == row['area_road']) &
                    (df_filtered['transfer_floor_level']
                     == row['transfer_floor_level']) &
                    (df_filtered['total_floors'] == row['total_floors']) &
                    (df_filtered['building_age'].between(
                        row['building_age'] - 0.5, row['building_age'] + 0.5)))
                subset = df_filtered.loc[mask, 'price']

                prices.append(
                    subset.median() if not subset.empty else np.nan)  # noqa
                # Calculate the mean of medians grouped by address
                # if not subset.empty and 'address' in df_filtered.columns:
                #     mean_of_medians = subset.groupby(
                #         'address').median().mean().values[0]
                # else:
                #     mean_of_medians = np.nan
                # prices.append(mxean_of_medians)
                # stds.append(subset.std() if not subset.empty else np.nan)
            result[f"other_similar_building_{year}_mean"] = prices
            # result[f"other_similar_house_{year}_std"] = stds
        X_copy = X_copy.assign(**result)
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class OtherSimilarHouseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.df = loader.load_input('other_prices.joblib')
        self.df = self.df[self.df['transfer_floor_level'] != 1]
        self._preprocess_data()
        self.df['year'] = self.df['transfer_date'].apply(get_transfer_year)
        self.df = self.df[self.df['year'].isin(
            ['2020', '2021', '2022', '2023'])]

    def _preprocess_data(self):
        self.df['remark'] = self.df['remark'].fillna('no_info')
        self.df = self.df[~self.df['remark'].str.contains('特殊')]
        self.df = self.df[~self.df['remark'].str.contains('攤位')]
        self.df = self.df[~self.df['remark'].str.contains('親友')]
        self.df = self.df[~self.df['remark'].str.contains('瑕疵')]
        self.df = self.df[~self.df['remark'].str.contains('身故')]
        self.df = self.df[self.df['transfer_floor_level'] != 1]
        self.df = self.df[self.df['building_type'].isin(
            ['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)', '公寓(5樓含以下無電梯)', '透天厝'])]
        self.df['price'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna(subset=['price'])
        self.df['price'] = self.df['price'].astype(int)
        self.df = self.df[self.df['price'] > 0]
        self.df['established_date'] = self.df['established_date'].apply(
            self.roc_to_ad)
        self.df['transfer_date'] = self.df['transfer_date'].apply(
            self.roc_to_ad)
        self.df['building_age'] = self.df.apply(lambda x: self.compute_age(
            x['transfer_date'], x['established_date']),
                                                axis=1)
        self.df['building_area'] = self.df['building_area'].clip(upper=1300)
        self.df['building_area'] = (self.df['building_area'] / 100) - 2
        self.df['building_area_rank'] = self.df['building_area'].rank(pct=True)

    def roc_to_ad(self, roc_date):
        if not isinstance(roc_date, str):
            return -1
        if roc_date == 'nan':
            return -1
        if roc_date[0] == "0":  # format: 0yyMMdd
            roc_year = int(roc_date[1:3])
        else:  # format: yyyMMdd
            try:
                roc_year = int(roc_date[:3])
            except ValueError:
                return -1
        if '.' in roc_date:
            roc_date = roc_date.split('.')[0]

        ad_year = 1911 + roc_year
        if ad_year > 2023:
            return -1
        ad_date = str(ad_year) + '-' + '01' + '-' + '01'
        return ad_date

    def compute_age(self, year, established_date):
        if established_date == -1:
            return -1
        year = pd.to_datetime(year)
        established_date = pd.to_datetime(established_date)
        days = (year - established_date).days
        days = days / 365
        return days

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        result = {}

        for year in self.df['year'].unique():
            df_filtered = self.df.loc[self.df['year'] == year].copy()
            df_filtered = df_filtered[df_filtered['building_age'] != -1]
            prices = []
            # stds = []
            for _, row in X_copy.iterrows():

                mask = (
                    (df_filtered['area_road'] == row['area_road']) &
                    (df_filtered['total_floors'] == row['total_floors']) &
                    (df_filtered['building_age'].between(
                        row['building_age'] - 0.5, row['building_age'] + 0.5)))
                subset = df_filtered.loc[mask, 'price']

                prices.append(
                    subset.median() if not subset.empty else np.nan)  # noqa
                # Calculate the mean of medians grouped by address
                # if not subset.empty and 'address' in df_filtered.columns:
                #     mean_of_medians = subset.groupby(
                #         'address').median().mean().values[0]
                # else:
                #     mean_of_medians = np.nan
                # prices.append(mxean_of_medians)
                # stds.append(subset.std() if not subset.empty else np.nan)
            result[f"other_similar_house_{year}_mean"] = prices
            # result[f"other_similar_house_{year}_std"] = stds
        X_copy = X_copy.assign(**result)
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class OtherHouseTransformerNew(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.df = loader.load_input('other_prices.joblib')
        self.df['remark'] = self.df['remark'].fillna('no_info')
        self.df = self.df[~self.df['remark'].str.contains('特殊')]
        self.df = self.df[~self.df['remark'].str.contains('攤位')]
        self.df = self.df[~self.df['remark'].str.contains('親友')]
        self.df = self.df[~self.df['remark'].str.contains('瑕疵')]
        self.df = self.df[~self.df['remark'].str.contains('身故')]
        # self.df = self.df[self.df['transfer_floor_level'] != 1]
        # self.df = self.df[self.df['building_type'].isin(
        #     ['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)', '公寓(5樓含以下無電梯)', '透天厝'])]
        self._preprocess_data()
        self.df['year'] = self.df['transfer_date'].apply(get_transfer_year)
        # self.df[self.df['year'].isin(['2020', '2021', '2022', '2023'])]
        self.get_price_map_new(self.df)

    def get_price_map_new(self, df):
        df = df[['area_road', 'total_floors', 'building_age', 'price']]
        grouped = df.groupby(['area_road', 'total_floors', 'building_age'])
        median_prices = grouped['price'].median().reset_index(
            name='median_price')
        grouped_again = median_prices.groupby(['area_road', 'building_age'])
        mean_of_medians = grouped_again['median_price'].mean().reset_index(
            name='mean_median_price')
        price_dict = pd.Series(mean_of_medians.mean_median_price.values,
                               index=mean_of_medians.set_index(
                                   ['area_road',
                                    'building_age']).index).to_dict()
        self.mean_map_total = price_dict

    def get_price_map(self, df):
        df = df[['area_road', 'building_age', 'price']]
        df = df.groupby(['area_road', 'building_age']).agg({
            'price': [
                ('price_mean', 'mean'),
            ]
        }).reset_index()
        df.columns = ['area_road', 'building_age', 'price_mean']
        self.mean_map_total = {
            (area_road, building_age): price
            for area_road, building_age, price in zip(
                df['area_road'], df['building_age'], df['price_mean'])
        }

    def _preprocess_data(self):
        self.df['price'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna(subset=['price'])
        self.df['price'] = self.df['price'].astype(int)
        self.df = self.df[self.df['price'] > 0]
        # self.df = self.df[~self.df['remark'].str.contains('特殊', na=False)]
        self.df['established_date'] = self.df['established_date'].apply(
            self.roc_to_ad)
        self.df['transfer_date'] = self.df['transfer_date'].apply(
            self.roc_to_ad)
        self.df['building_age'] = self.df.apply(lambda x: self.compute_age(
            x['transfer_date'], x['established_date']),
                                                axis=1)
        self.df['building_age'] = self.df['building_age'].apply(
            self.binning_age)

    def roc_to_ad(self, roc_date):
        if not isinstance(roc_date, str):
            return -1
        if roc_date == 'nan':
            return -1
        if roc_date[0] == "0":  # format: 0yyMMdd
            roc_year = int(roc_date[1:3])
        else:  # format: yyyMMdd
            try:
                roc_year = int(roc_date[:3])
            except ValueError:
                return -1
        if '.' in roc_date:
            roc_date = roc_date.split('.')[0]

        ad_year = 1911 + roc_year
        if ad_year > 2023:
            return -1
        ad_date = str(ad_year) + '-' + '01' + '-' + '01'
        return ad_date

    def compute_age(self, year, established_date):
        if established_date == -1:
            return -1
        year = pd.to_datetime(year)
        established_date = pd.to_datetime(established_date)
        days = (year - established_date).days
        days = days / 365
        return days

    def binning_age(self, age):
        if age <= -1:
            return -1
        if age < 10:
            return 0
        elif age < 20:
            return 1
        elif age < 30:
            return 2
        elif age < 40:
            return 3
        elif age < 50:
            return 4
        else:
            return 5

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        using_ages = X_copy['building_age'].apply(self.binning_age)
        using_roads = X_copy['area_road']
        r_ = []
        for area_road, b_age in zip(using_roads, using_ages):
            try:
                r_.append(self.mean_map_total[(area_road, b_age)])
            except KeyError:
                r_.append(np.nan)
        X_copy.loc[:, 'other_price_mean_age'] = r_

        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class OtherHouseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.df = loader.load_input('other_prices.joblib')
        self.df['remark'] = self.df['remark'].fillna('no_info')
        self.df = self.df[~self.df['remark'].str.contains('特殊')]
        self.df = self.df[~self.df['remark'].str.contains('攤位')]
        self.df = self.df[~self.df['remark'].str.contains('親友')]
        self.df = self.df[~self.df['remark'].str.contains('瑕疵')]
        self.df = self.df[~self.df['remark'].str.contains('身故')]
        self.df = self.df[~self.df['remark'].str.contains('預售屋')]
        self._preprocess_data()
        self.df['year'] = self.df['transfer_date'].apply(get_transfer_year)
        self.df['year'] = self.df['transfer_date'].apply(get_transfer_year)
        self.df[self.df['year'].isin(['2020', '2021', '2022', '2023'])]
        self.get_price_map_new(self.df)

    def get_price_map_new(self, df):
        df = df[['area_road', 'total_floors', 'building_age', 'price']]
        grouped = df.groupby(['area_road', 'total_floors', 'building_age'])
        median_prices = grouped['price'].median().reset_index(
            name='median_price')
        grouped_again = median_prices.groupby(['area_road'])
        mean_of_medians = grouped_again['median_price'].mean().reset_index(
            name='mean_median_price')
        price_dict = pd.Series(mean_of_medians.mean_median_price.values,
                               index=mean_of_medians.set_index(
                                   ['area_road']).index).to_dict()
        self.mean_map_total = price_dict

    def get_price_map(self, df):
        df = df[['area_road', 'price']]
        df = df.groupby('area_road').agg({
            'price': [('price_mean', 'mean'), ('price_std', 'std'),
                      ('price_count', 'count'), ('price_median', 'median'),
                      ('price_max', 'max'), ('price_min', 'min'),
                      ('price_top_20_mean', self.top_20_mean),
                      ('price_middle_40_mean', self.middle_40_mean),
                      ('price_bottom_40_mean', self.bottom_40_mean)]
        }).reset_index()
        df.columns = [
            'area_road', 'price_mean', 'price_std', 'price_count',
            'price_median', 'price_max', 'price_min', 'price_top_20_mean',
            'price_middle_40_mean', 'price_bottom_40_mean'
        ]
        self.mean_map_total = {
            area_road: price
            for area_road, price in zip(df['area_road'], df['price_mean'])
        }

    def _preprocess_data(self):
        self.df['price'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna(subset=['price'])
        self.df['price'] = self.df['price'].astype(int)
        self.df = self.df[self.df['price'] > 0]
        self.df['established_date'] = self.df['established_date'].apply(
            self.roc_to_ad)
        self.df['transfer_date'] = self.df['transfer_date'].apply(
            self.roc_to_ad)
        self.df['building_age'] = self.df.apply(lambda x: self.compute_age(
            x['transfer_date'], x['established_date']),
                                                axis=1)
        self.df['building_age'] = self.df['building_age'].apply(
            self.binning_age)

    def roc_to_ad(self, roc_date):
        if not isinstance(roc_date, str):
            return -1
        if roc_date == 'nan':
            return -1
        if roc_date[0] == "0":  # format: 0yyMMdd
            roc_year = int(roc_date[1:3])
        else:  # format: yyyMMdd
            try:
                roc_year = int(roc_date[:3])
            except ValueError:
                return -1
        if '.' in roc_date:
            roc_date = roc_date.split('.')[0]

        ad_year = 1911 + roc_year
        if ad_year > 2023:
            return -1
        ad_date = str(ad_year) + '-' + '01' + '-' + '01'
        return ad_date

    def compute_age(self, year, established_date):
        if established_date == -1:
            return -1
        year = pd.to_datetime(year)
        established_date = pd.to_datetime(established_date)
        days = (year - established_date).days
        days = days / 365
        return days

    def binning_age(self, age):
        if age == -1:
            return -1
        if age < 10:
            return 0
        elif age < 20:
            return 1
        elif age < 30:
            return 2
        elif age < 40:
            return 3
        elif age < 50:
            return 4
        else:
            return 5

    def fit(self, X, y=None, **fit_params):

        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        X_copy.loc[:, 'other_price_mean'] = X_copy['area_road'].map(
            self.mean_map_total)

        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class PostPreprocessTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, threshold: float):
        self.threshold = threshold

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        est_ratio = 1.6e-05
        X_copy = X.copy()
        X_copy['other_price_mean'] = X_copy['other_price_mean'].replace(
            -1, np.nan)
        X_copy['other_price_mean_age'] = X_copy[
            'other_price_mean_age'].replace(-1, np.nan)
        X_copy['building_age_ratio'] = X_copy['building_age'] / X_copy[
            'building_age_area_{}_mean'.format(self.threshold)]
        X_copy['building_age_area_1000_mean'] = X_copy[
            'building_age'] / X_copy['building_age_area_1000_mean']
        X_copy['y_area_{}_other_price'.format(
            self.threshold)] = X_copy['y_area_{}_mean'.format(
                self.threshold)] / X_copy[  # noqa
                    'other_price_mean']
        X_copy['y_area_{}_other_price'.format(
            self.threshold)] = X_copy['y_area_{}_other_price'.format(
                self.threshold)] / est_ratio
        X_copy['y_area_{}_different_group'.format(
            self.threshold)] = X_copy['y_area_{}_mean'.format(
                self.threshold)] / X_copy['y_area_1000_mean']
        X_copy['y_area_{}_mean'.format(self.threshold)] / X_copy[  # noqa
            'other_price_mean']
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)


class DistMeanTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 threshold: float,
                 groupby_name: str,
                 target_name: str = None):
        self.threshold = threshold
        self.groupby_name = groupby_name
        self.target_name = target_name
        if self.target_name is None:
            the_name = 'y'
        else:
            the_name = self.target_name
        self.output_names = [
            "{}_area_{}_mean".format(the_name, threshold),
        ]
        self.group_info = {}

    def compute_single_to_group(self, h_id, lon1, lat1, group):
        try:
            a_group = self.group_info[group].copy()
        except KeyError:
            return np.nan
        group_use = a_group.copy()
        dists = haversine_distance_vectorized(lat1, lon1, group_use['lat'],
                                              group_use['lon'])
        group_use['dist'] = dists
        group_use['dist'] = group_use['dist'] * 1000
        group_use = group_use[(group_use['dist'] < self.threshold)
                              & (group_use['id'] != h_id)]
        if len(group_use) == 0:
            return np.nan
        if self.target_name is None:
            group_mean = group_use['y'].mean()
        else:
            group_mean = group_use[self.target_name].mean()
        return group_mean

    def fit(self, X, y=None, **fit_params):
        X_use = X.copy()
        y_use = y.copy()
        if self.target_name is not None:
            X_use = X_use[[
                'id', 'lon', 'lat', self.groupby_name, self.target_name
            ]]
        else:
            X_use = X_use[['id', 'lon', 'lat', self.groupby_name]]
        X_use = pd.concat([X_use, y_use], axis=1)
        X_group = X_use.groupby(self.groupby_name)
        X_group = {df[0]: df[1] for df in X_group}
        self.group_info = X_group
        return self

    def transform(self, X, y=None, **fit_params):
        X_use = X.copy()
        outputs_mean = []
        for h_id, lon1, lat1, group in zip(X_use['id'], X_use['lon'],
                                           X_use['lat'],
                                           X_use[self.groupby_name]):
            output_mean = self.compute_single_to_group(h_id, lon1, lat1, group)
            outputs_mean.append(output_mean)
        for output_name in self.output_names:
            if output_name.endswith("mean"):
                X_use[output_name] = outputs_mean
        return X_use

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class UniversityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.university = loader.load_train_data('external_data/大學基本資料.csv',
                                                 data_type='csv')
        self.output_1 = "university_dist"
        self.output_2 = "university_name"

    def compute_a_single(self, lat1, lon1):
        university = self.university.copy()
        dists = haversine_distance_vectorized(lat1, lon1, university['lat'],
                                              university['lng'])
        university['dist'] = dists
        min_dist_idx = university['dist'].idxmin()
        nearest_university = university.loc[min_dist_idx, '學校名稱']
        return nearest_university, university.loc[min_dist_idx, 'dist']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        schools = []
        dists = []
        for lat, lng in zip(X_copy['lat'], X_copy['lon']):
            school, dist = self.compute_a_single(lat, lng)
            schools.append(school)
            dists.append(dist)
        X_copy[self.output_1] = dists
        X_copy[self.output_2] = schools
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class SeniorHighSchoolTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.senior_high_school = loader.load_train_data(
            'external_data/國中基本資料.csv', data_type='csv')
        self.output_1 = "senior_high_school_dist"
        self.output_2 = "senior_high_school_name"

    def compute_a_single(self, lat1, lon1):
        senior_high_school = self.senior_high_school.copy()
        dists = haversine_distance_vectorized(lat1, lon1,
                                              senior_high_school['lat'],
                                              senior_high_school['lng'])
        senior_high_school['dist'] = dists
        min_dist_idx = senior_high_school['dist'].idxmin()
        nearest_senior_high_school = senior_high_school.loc[min_dist_idx,
                                                            '學校名稱']
        return nearest_senior_high_school, senior_high_school.loc[min_dist_idx,
                                                                  'dist']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        schools = []
        dists = []
        for lat, lng in zip(X_copy['lat'], X_copy['lon']):
            school, dist = self.compute_a_single(lat, lng)
            schools.append(school)
            dists.append(dist)
        X_copy[self.output_1] = dists
        X_copy[self.output_2] = schools
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class HighSchoolTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.high_school = loader.load_train_data('external_data/高中基本資料.csv',
                                                  data_type='csv')
        self.output_1 = "high_school_dist"
        self.output_2 = "high_school_name"

    def compute_a_single(self, lat1, lon1):
        high_school = self.high_school.copy()
        dists = haversine_distance_vectorized(lat1, lon1, high_school['lat'],
                                              high_school['lng'])
        high_school['dist'] = dists
        min_dist_idx = high_school['dist'].idxmin()
        nearest_high_school = high_school.loc[min_dist_idx, '學校名稱']
        return nearest_high_school, high_school.loc[min_dist_idx, 'dist']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        schools = []
        dists = []
        for lat, lng in zip(X_copy['lat'], X_copy['lon']):
            school, dist = self.compute_a_single(lat, lng)
            schools.append(school)
            dists.append(dist)
        X_copy[self.output_1] = dists
        X_copy[self.output_2] = schools
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class HospitalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, threshold: int = 100):
        loader = dataloader.DataLoader()
        self.hospital = loader.load_train_data('external_data/醫療機構基本資料.csv',
                                               data_type='csv')
        self.hospital = self.hospital[self.hospital['醫師'] > threshold]
        self.output_1 = "hospital_dist"
        self.output_2 = "hospital_name"

    def compute_a_single(self, lat1, lon1):
        hospital = self.hospital.copy()
        dists = haversine_distance_vectorized(lat1, lon1, hospital['lat'],
                                              hospital['lng'])
        hospital['dist'] = dists
        min_dist_idx = hospital['dist'].idxmin()
        nearest_hospital = hospital.loc[min_dist_idx, '機構名稱']
        return nearest_hospital, hospital.loc[min_dist_idx, 'dist']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        stations = []
        dists = []
        for lat, lng in zip(X_copy['lat'], X_copy['lon']):
            station, dist = self.compute_a_single(lat, lng)
            stations.append(station)
            dists.append(dist)
        X_copy[self.output_1] = dists
        X_copy[self.output_2] = stations
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class TrainTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        loader = dataloader.DataLoader()
        self.train_station = loader.load_train_data('external_data/火車站點資料.csv',
                                                    data_type='csv')
        self.output_1 = "train_dist"
        self.output_2 = "train_name"

    def haversine_distance_vectorized(self, lat1, lon1, lat2, lon2):
        R = 6371

        d_lat = np.radians(lat2 - lat1)
        d_lon = np.radians(lon2 - lon1)

        a = (np.sin(d_lat / 2)**2 + np.cos(np.radians(lat1)) *
             np.cos(np.radians(lat2)) * np.sin(d_lon / 2)**2)

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        return distance

    def compute_a_single(self, lat1, lon1):
        train_copy = self.train_station.copy()
        dists = self.haversine_distance_vectorized(lat1, lon1,
                                                   train_copy['lat'],
                                                   train_copy['lng'])
        train_copy['dist'] = dists
        min_dist_idx = train_copy['dist'].idxmin()
        nearest_station = train_copy.loc[min_dist_idx, '站點名稱']
        return nearest_station, train_copy.loc[min_dist_idx, 'dist']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        stations = []
        dists = []
        for lat, lng in zip(X_copy['lat'], X_copy['lon']):
            station, dist = self.compute_a_single(lat, lng)
            stations.append(station)
            dists.append(dist)
        X_copy[self.output_1] = dists
        X_copy[self.output_2] = stations
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class MRTTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, restricted_city: list, mrt_type: str):
        loader = dataloader.DataLoader()
        mrt_station = loader.load_train_data('external_data/捷運站點資料.csv',
                                             data_type='csv')
        self.mrt_station = mrt_station[mrt_station['站點UID'].str.contains(
            mrt_type)].copy()
        self.restricted_city = restricted_city
        self.output_1 = "{}_mrt_dist".format(mrt_type)
        self.output_2 = "{}_mrt_name".format(mrt_type)

    def haversine_distance_vectorized(self, lat1, lon1, lat2, lon2):
        R = 6371  # 地球的半徑 (公里)

        d_lat = np.radians(lat2 - lat1)
        d_lon = np.radians(lon2 - lon1)

        a = (np.sin(d_lat / 2)**2 + np.cos(np.radians(lat1)) *
             np.cos(np.radians(lat2)) * np.sin(d_lon / 2)**2)

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        return distance

    def compute_a_single(self, lat1, lon1):
        mrt_copy = self.mrt_station.copy()
        dists = self.haversine_distance_vectorized(lat1, lon1, mrt_copy['lat'],
                                                   mrt_copy['lng'])
        mrt_copy['dist'] = dists
        min_dist_idx = mrt_copy['dist'].idxmin()
        nearest_station = mrt_copy.loc[min_dist_idx, '站點名稱']
        return nearest_station, mrt_copy.loc[min_dist_idx, 'dist']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        stations = []
        dists = []
        X_use = X_copy[X_copy['city'].isin(self.restricted_city)].copy()
        for lat, lng in zip(X_use['lat'], X_use['lon']):
            station, dist = self.compute_a_single(lat, lng)
            stations.append(station)
            dists.append(dist)
        X_use[self.output_1] = dists
        X_use[self.output_2] = stations
        X_copy = pd.merge(X_copy,
                          X_use[['id', self.output_1, self.output_2]],
                          on='id',
                          how='left')
        X_copy[self.output_1].fillna(np.nan, inplace=True)
        X_copy[self.output_2].fillna('no_info', inplace=True)
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class MRTAllTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.transformers = (
            MRTTransformer(restricted_city=['新北市'], mrt_type='NTDLRT'),
            MRTTransformer(restricted_city=['高雄市'], mrt_type='KRTC'),
            MRTTransformer(restricted_city=["台北市", "新北市", '桃園市'],
                           mrt_type='TYMC'),
            MRTTransformer(restricted_city=['高雄市'], mrt_type='KLRT'),
            MRTTransformer(restricted_city=["台北市", "新北市"], mrt_type='TRTC'),
        )

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        for transformer in self.transformers:
            X_copy = transformer.fit_transform(X_copy)
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class EncoderDict(UserDict):

    def __init__(self,
                 *args,
                 encoder_name="no_info",
                 default_vale=0,
                 debug=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.name = encoder_name
        self.default_value = default_vale

    def __missing__(self, key):
        if self.debug:
            print(
                "Fail to encode the feature {} with the key {} fillna with {}".
                format(self.name, key, self.default_value))
        return self.default_value


class LabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, var_name: str):
        self.var_name = var_name
        self.encoder = EncoderDict(
            encoder_name='{}_label_encoder'.format(var_name),
            default_vale=np.nan)

    def fit(self, X, y=None, **fit_params):
        X_use = X.copy()
        unique_values = X_use[self.var_name].unique()
        for i, value in enumerate(unique_values):
            self.encoder.update({value: i + 1})
        return self

    def transform(self, X, y=None, **fit_params):
        X_use = X.copy()
        values = list(X_use[self.var_name].map(self.encoder))
        return values

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)


class TargetVarianceEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, var_name: str, tar_name: str):
        self.var_name = var_name
        self.tar_name = tar_name
        self.output_name = "{}_var".format(var_name)
        self.encoder = EncoderDict(
            encoder_name="{}_var_encoder".format(var_name),
            default_vale=np.nan,
        )

    def fit(self, X, y=None, **fit_params):
        tmp = pd.concat([X[[self.var_name]].copy(), y.copy()], axis=1)
        tmp = tmp[[self.var_name, self.tar_name]].groupby(
            self.var_name)[self.tar_name].var().reset_index()
        for var, tar in zip(tmp[self.var_name], tmp[self.tar_name]):
            self.encoder.update({var: tar})
        return self

    def transform(self, X, y=None, **fit_params):
        X_use = X.copy()
        X_use[self.output_name] = X_use[self.var_name].map(self.encoder)
        return X_use

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)


class TargetMeanEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, var_name: str, tar_name: str):
        self.var_name = var_name
        self.tar_name = tar_name
        self.output_name = "{}_mean".format(var_name)
        self.encoder = EncoderDict(
            encoder_name="{}_mean_encoder".format(var_name),
            default_vale=np.nan,
        )

    def fit(self, X, y=None, **fit_params):
        tmp = pd.concat([X[[self.var_name]].copy(), y.copy()], axis=1)
        tmp = tmp[[self.var_name, self.tar_name]].groupby(
            self.var_name)[self.tar_name].mean().reset_index()
        for var, tar in zip(tmp[self.var_name], tmp[self.tar_name]):
            self.encoder.update({var: tar})
        return self

    def transform(self, X, y=None, **fit_params):
        X_use = X.copy()
        X_use[self.output_name] = X_use[self.var_name].map(self.encoder)
        return X_use

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)


class TargetCountEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, var_name: str, tar_name: str):
        self.var_name = var_name
        self.tar_name = tar_name
        self.output_name = "{}_count".format(var_name)
        self.encoder = EncoderDict(
            encoder_name="{}_count_encoder".format(var_name))

    def fit(self, X, y=None, **fit_params):
        tmp = pd.concat([X[[self.var_name]].copy(), y.copy()], axis=1)
        tmp = tmp[[self.var_name, self.tar_name]].groupby(
            self.var_name)[self.tar_name].count().reset_index()
        tmp[self.tar_name] = tmp[self.tar_name] / tmp[self.tar_name].sum()
        for var, tar in zip(tmp[self.var_name], tmp[self.tar_name]):
            self.encoder.update({var: tar})
        return self

    def transform(self, X, y=None, **fit_params):
        X_use = X.copy()
        X_use[self.output_name] = X_use[self.var_name].map(self.encoder)
        return X_use

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)


class RawFeatExtracter(BaseEstimator, TransformerMixin):

    def instantiate(self, param):
        if isinstance(param, str):
            return [param]
        elif isinstance(param, list):
            return param
        else:
            return None

    def __init__(self,
                 numeric=None,
                 cat=None,
                 target_mean=None,
                 target_count=None,
                 target_var=None,
                 remove_cols=None):
        if cat is not None:
            self.cat = self.instantiate(cat)
        if numeric is not None:
            self.numeric = self.instantiate(numeric)
        if target_mean is not None:
            self.target_mean = self.instantiate(target_mean)
        if target_count is not None:
            self.target_count = self.instantiate(target_count)
        if target_var is not None:
            self.target_var = self.instantiate(target_var)
        if remove_cols is not None:
            self.remove_cols = self.instantiate(remove_cols)
        self.cat_encoders = {}
        self.target_mean_encoders = {}
        self.target_count_encoders = {}
        self.target_var_encoders = {}

    def fit(self, X, y=None, **fit_params):
        # Fit label encoders for categorical columns
        if self.cat:
            for feature in self.cat:
                le = LabelEncoder(var_name=feature)
                le.fit(X[[feature]])
                self.cat_encoders[feature] = le
        if self.target_count:
            for feature in self.target_count:
                self.target_count_encoders[feature] = TargetCountEncoder(
                    feature, "y")
                self.target_count_encoders[feature].fit(X, y)
        if self.target_mean:
            for feature in self.target_mean:
                self.target_mean_encoders[feature] = TargetMeanEncoder(
                    feature, "y")
                self.target_mean_encoders[feature].fit(X, y)
        if self.target_var:
            for feature in self.target_var:
                self.target_var_encoders[feature] = TargetVarianceEncoder(
                    feature, "y")
                self.target_var_encoders[feature].fit(X, y)
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()

        # Target mean encoding
        if self.target_mean:
            for feature in self.target_mean:
                encoder = self.target_mean_encoders[feature]
                X_copy = encoder.transform(X_copy)

        # Target count encoding
        if self.target_count:
            for feature in self.target_count:
                encoder = self.target_count_encoders[feature]
                X_copy = encoder.transform(X_copy)

        # Target variance encoding
        if self.target_var:
            for feature in self.target_var:
                encoder = self.target_var_encoders[feature]
                X_copy = encoder.transform(X_copy)

        # Label encoding
        if self.cat:
            for feature in self.cat:
                encoder = self.cat_encoders[feature]
                X_copy[feature] = encoder.transform(X_copy[[feature]])

        # remove columns
        if self.remove_cols:
            for col in self.remove_cols:
                try:
                    X_copy.drop(col, axis=1, inplace=True)
                except KeyError:
                    print("Fail to remove the column {}".format(col))
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
