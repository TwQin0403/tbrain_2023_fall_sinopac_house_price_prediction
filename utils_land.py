import zipfile
import io
import pandas as pd
import re
import glob
import os
import dataloader
import etl_tool as e_tool

city_code_dict = {
    'a': "台北市",
    'b': "台中市",
    'c': "基隆市",
    'd': "台南市",
    'e': "高雄市",
    'f': "新北市",
    'g': "宜蘭縣",
    'h': "桃園市",
    'j': "新竹縣",
    'k': "苗栗縣",
    'l': "台中市",
    'm': "南投縣",
    'n': "彰化縣",
    'p': "雲林縣",
    'q': "嘉義縣",
    'r': "台南市",
    's': "高雄市",
    't': "屏東縣",
    'u': "花蓮縣",
    'v': "臺東縣",
    'x': "澎湖縣",
    'y': "陽明山",
    'w': "金門縣",
    'z': "連江縣",
    'i': "嘉義市",
    'o': "新竹市",
}


def get_train_area_road_geo():
    loader = dataloader.DataLoader()
    train = loader.load_train_data('training_data.csv', data_type='csv')
    test = loader.load_train_data('public_dataset.csv', data_type='csv')
    train = e_tool.preprocess_raw_data(train, is_simple=True)
    test = e_tool.preprocess_raw_data(test, is_simple=True)
    test.loc[:, 'y'] = -1
    df = pd.concat([train, test]).reset_index(drop=True)
    df = df[['lat', 'lon',
             'area_road']].groupby('area_road').mean().reset_index()
    mapping = {
        area_road: (lat, lon)
        for area_road, lat, lon in zip(df['area_road'], df['lat'], df['lon'])
    }
    return mapping


def merge_to_table(results):
    df = pd.DataFrame()
    for roc_date in results.keys():
        date = update_date(roc_date)
        for city in results[roc_date].keys():
            tmp_df = results[roc_date][city]
            tmp_df['date'] = date
            tmp_df['city'] = city
            df = df.append(tmp_df)
    return df


def fetch_all_zip_file(fetch_type='buy'):
    loader = dataloader.DataLoader()
    house_data_path = loader.input_path
    subdirectories = os.walk(house_data_path)
    subdirectories = [x[0] for x in subdirectories][1:]
    results = {}
    for subdir in subdirectories:
        results.update(fetch_single_zip_folder(subdir, fetch_type))
    return results


def fetch_single_zip_file(zipfile_path, fetch_type='buy'):
    archive = zipfile.ZipFile(zipfile_path, 'r')
    filelist = archive.filelist
    tables = {}
    for a_file in filelist:
        filename = a_file.filename
        if filename.split('.')[-1] == 'csv':
            city_code = filename.split('_')[0]
            if len(city_code) > 1:
                continue
            else:
                city_name = city_code_dict[city_code]

            regex = r".+_(.*)\.csv"
            matches = re.findall(regex, filename)[0]
            if fetch_type == 'buy':
                match_cond = 'a'
            elif fetch_type == 'rent':
                match_cond = 'c'
            else:
                match_cond = fetch_type
            if matches == match_cond:
                df = pd.read_csv(io.BytesIO(archive.read(a_file.filename)),
                                 error_bad_lines=False)
                tables.update({city_name: df})
    return tables


def fetch_single_zip_folder(folder_path, fetch_type='buy'):
    os.chdir(folder_path)
    zip_file_paths = glob.glob('*.zip')
    result = {}
    for zip_file_path in zip_file_paths:
        table = fetch_single_zip_file(zip_file_path, fetch_type)
        key = zip_file_path.replace('.zip', '')
        result.update({key: table})
    return result


def change_names_zip_folder(folder_path):
    year = folder_path.split('\\')[-1]
    mappings = {
        "lvr_landcsv.zip": year + '01',
        "lvr_landcsv(1).zip": year + '02',
        "lvr_landcsv(2).zip": year + '03',
        "lvr_landcsv(3).zip": year + '04',
    }
    os.chdir(folder_path)
    files = os.listdir(folder_path)
    for file_name in files:
        os.rename(os.path.join(folder_path, file_name),
                  os.path.join(folder_path, mappings[file_name] + ".zip"))


def update_date_day(roc_date):
    roc_date = str(roc_date)
    roc_year = roc_date[:3]
    date = roc_date[3:]
    use_year = str(int(roc_year) + 1911)
    use_month = date[:2]
    use_date = date[2:]
    return use_year + '-' + use_month + '-' + use_date


def update_date(roc_date):
    season_dict = {
        "01": "01-01",
        "02": "04-01",
        "03": "07-01",
        "04": "10-01",
    }
    roc_year = roc_date[:3]
    season = roc_date[3:]
    use_year = str(int(roc_year) + 1911)
    use_date = season_dict[season]
    return use_year + "-" + use_date


def merge_results(results):
    use_cities = [
        '台北市', '高雄市', '新北市', '桃園市', '台中市', '台南市', '苗栗縣', '新竹縣', '基隆市', '屏東縣',
        '新竹市', '宜蘭縣', '花蓮縣', '嘉義市', '金門縣', '嘉義縣', '彰化縣', '雲林縣'
    ]
    m_results = []
    for year_season in results.keys():
        year = str(int(year_season[:3]) + 1911)
        season = year_season[3:]
        a_data = results[year_season]
        for city in use_cities:
            df = a_data[city]
            df.loc[:, 'city'] = city
            df.loc[:, 'year'] = year
            df.loc[:, 'season'] = season
            m_results.append(df)
    return pd.concat(m_results).reset_index(drop=True)


def roc_to_ad(roc_date):
    if not isinstance(roc_date, str):
        return None
    if roc_date[0] == "0":  # format: 0yyMMdd
        roc_year = int(roc_date[1:3])
    else:  # format: yyyMMdd
        roc_year = int(roc_date[:3])

    ad_year = 1911 + roc_year
    if ad_year > 2020:
        return None
    ad_date = str(ad_year) + '-' + '01' + '-' + '01'
    return ad_date


def clean_address(address):
    # Regex pattern for detecting street information
    street_pattern = re.compile(r".*?[市|縣].*?[區|鎮|鄉].*?[路|街|巷|道|弄]")

    # Search for the pattern in the address
    match = street_pattern.search(address)

    # If a match is found, return it; otherwise, return the original address
    return match.group() if match else address


def convert_chinese_numbers_to_integers(chinese_numbers_column):
    chinese_to_arabic_mapping = {
        '一': 1,
        '二': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9,
        '十': 10
    }

    def chinese_to_arabic(chinese_number):
        # 處理 'nan'
        if pd.isna(chinese_number):
            return -1  # 如果是nan返回-1

        if chinese_number == '全':
            return -2

        chinese_number = chinese_number.replace('夾層，', '')
        chinese_number = chinese_number.replace('夾層', '')
        if len(chinese_number) == 0:
            return -1

        if '，' in chinese_number:
            chinese_number = chinese_number.split('，')[0]

        # 去除 '層' 並分割字符串
        chinese_number = chinese_number.rstrip('層')

        # 處理一位數和 '十'
        if len(chinese_number) == 1:
            return chinese_to_arabic_mapping[chinese_number]

        # 處理十位以上的數字
        if chinese_number.startswith('十'):
            if len(chinese_number) == 1:
                return 10
            else:
                return 10 + chinese_to_arabic_mapping[chinese_number[1]]
        elif chinese_number.endswith('十'):
            return chinese_to_arabic_mapping[chinese_number[0]] * 10
        elif chinese_number.startswith('十') and chinese_number.endswith('十'):
            ones = chinese_number.split('十')
            return 10 + chinese_to_arabic_mapping[ones]
        else:
            # 兩位數，例如：二十一
            try:
                tens, ones = chinese_number.split('十')
                return chinese_to_arabic_mapping[
                    tens] * 10 + chinese_to_arabic_mapping[ones]
            except Exception:
                print(chinese_number)
                return -1

    # 轉換列中的每個元素
    return [chinese_to_arabic(number) for number in chinese_numbers_column]


def processing_land():
    loader = dataloader.DataLoader()
    train = loader.load_train_data('training_data.csv', data_type='csv')
    public = loader.load_train_data('public_dataset.csv', data_type='csv')
    private = loader.load_train_data('private_dataset.csv', data_type='csv')
    test = pd.concat([public, private]).reset_index(drop=True)

    train = e_tool.preprocess_raw_data(train, is_simple=True)
    test = e_tool.preprocess_raw_data(test, is_simple=True)
    train = train[['lat', 'lon', 'city', 'area', 'area_road']]
    test = test[['lat', 'lon', 'city', 'area', 'area_road']]
    data = pd.concat([train, test]).reset_index(drop=True)
    all_area_road = list(data['area_road'].unique())
    all_area = list(data['area'].unique())
    results = fetch_all_zip_file()
    df = merge_results(results)
    df['土地位置建物門牌'] = df['土地位置建物門牌'].apply(lambda x: x.replace('臺北市', '台北市'))
    df['土地位置建物門牌'] = df['土地位置建物門牌'].apply(lambda x: x.replace('臺中市', '台中市'))
    df['土地位置建物門牌'] = df['土地位置建物門牌'].apply(lambda x: x.replace('臺南市', '台南市'))
    df = df[df['交易標的'].isin(['房地(土地+建物)', '房地(土地+建物)+車位', '建物'])]
    # # 處理車價
    df = df.dropna(subset=['單價元平方公尺'])
    df['車位移轉總面積(平方公尺)'] = df['車位移轉總面積(平方公尺)'].fillna(0)
    df['建物移轉總面積平方公尺'] = df['建物移轉總面積平方公尺'].astype(float)
    df['車位移轉總面積(平方公尺)'] = df['車位移轉總面積(平方公尺)'].astype(float)
    df['車位總價元'] = df['車位總價元'].astype(float)
    df['總價元'] = df['總價元'].astype(float)
    mask = (df['車位移轉總面積(平方公尺)'] != 0) & (df['車位總價元'] > 0) & (
        df['建物移轉總面積平方公尺'] - df['車位移轉總面積(平方公尺)'] != 0)
    df.loc[mask, '單價元平方公尺'] = ((df['總價元'] - df['車位總價元']) /
                               (df['建物移轉總面積平方公尺'] - df['車位移轉總面積(平方公尺)']))
    df = df[[
        '土地位置建物門牌', '單價元平方公尺', '建築完成年月', '備註', '總樓層數', '移轉層次', '建物型態', '主要用途',
        '主要建材', '建物移轉總面積平方公尺', 'city', 'year', 'season', '交易年月日'
    ]]

    def check_area_road(s):
        for area in all_area_road:
            if area in s:
                return area
        return None

    def check_area(s):
        for area in all_area:
            if area in s:
                return area
        return None

    df['建築完成年月'] = df['建築完成年月'].astype(str)
    df['area_road'] = df['土地位置建物門牌'].apply(check_area_road)
    df['area'] = df['土地位置建物門牌'].apply(check_area)
    df = df.dropna(subset=['area_road'])
    df.columns = [
        'address', 'price', 'established_date', 'remark', 'total_floors',
        'transfer_floor_level', 'building_type', 'primary_use',
        'main_material', 'building_area', 'city', 'year', 'season',
        'transfer_date', 'area_road', 'area'
    ]

    df['total_floors'] = convert_chinese_numbers_to_integers(
        df['total_floors'].to_list())
    df['transfer_floor_level'] = convert_chinese_numbers_to_integers(
        df['transfer_floor_level'].to_list())
    df = df.reset_index(drop=True)
    loader.save_input(df, 'other_prices.joblib')
    return df
