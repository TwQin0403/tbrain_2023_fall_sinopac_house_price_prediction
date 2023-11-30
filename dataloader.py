import pandas as pd
import os
import joblib
from pathlib import Path


class DataLoader():

    def __init__(self):
        file_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.file_path = file_path
        self.input_path = file_path / 'input'
        self.train_path = file_path / 'train_data'
        self.model_path = file_path / 'model'

    def __repr__(self):
        return "DataLoader(file_path = {})".format(self.file_path)

    def __str__(self):
        return "DataLoader(file_path = {})".format(self.file_path)

    def _load(self, file_path, data_type='joblib', **kwargs):
        if data_type == 'joblib':
            data = joblib.load(file_path, **kwargs)
        elif data_type == 'csv':
            data = pd.read_csv(file_path, **kwargs)
        elif data_type == 'excel':
            data = pd.read_excel(file_path, **kwargs)
        return data

    def _save(self, cls, file_path, cls_type='joblib', **kwargs):
        if cls_type == 'csv':
            cls.to_csv(file_path, index=None, **kwargs)
        else:
            joblib.dump(cls, file_path, **kwargs)

    def load_input(self, data_name, data_type='joblib', **kwargs):
        file_path = self.input_path / data_name
        return self._load(file_path, data_type, **kwargs)

    def load_train_data(self, data_name, data_type='joblib', **kwargs):
        file_path = self.train_path / data_name
        return self._load(file_path, data_type, **kwargs)

    def load_model(self, data_name, data_type='joblib', **kwargs):
        file_path = self.model_path / data_name
        return self._load(file_path, data_type, **kwargs)

    def save_input(self, cls, data_name, cls_type='joblib', **kwargs):
        file_path = self.input_path / data_name
        self._save(cls, file_path, cls_type, **kwargs)

    def save_train_data(self, cls, data_name, cls_type='joblib', **kwargs):
        file_path = self.train_path / data_name
        self._save(cls, file_path, cls_type, **kwargs)

    def save_model(self, cls, data_name, cls_type='joblib', **kwargs):
        file_path = self.model_path / data_name
        self._save(cls, file_path, cls_type, **kwargs)
