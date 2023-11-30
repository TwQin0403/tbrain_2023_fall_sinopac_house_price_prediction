import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
# from sklearn.cluster import KMeans
import dataloader
import etl_config as e_config
import etl_tool as e_tool
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

LOADER = dataloader.DataLoader()


def lgb_mape(preds, train_data):
    y_true = train_data.get_label()
    grad = -100 * (y_true - preds) / (y_true * (np.abs(y_true - preds) + 1))
    hess = 100 / (y_true * (np.abs(y_true - preds) + 1)**2)
    return grad, hess


def lgb_mape_eval(preds, train_data):
    y_true = train_data.get_label()
    mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
    return 'MAPE', mape, False


def replace_transformer(transformer, mask_cols: list, X, y=None):
    X_copy = X.copy()
    X_copy = X_copy.drop(columns=mask_cols)
    X_copy = transformer.fit_transform(X_copy, y)
    return X_copy


class LGBTrainer():

    def __init__(self, n_folds=10):
        self.trainer = 'lgb'
        self.fold_handler = KFold(n_splits=n_folds,
                                  shuffle=True,
                                  random_state=42)
        self.set_config()
        self.models = []
        self.predictions = []  # 存放每一折的預測結果
        self.residuals = []  # 存放每一折的殘差
        self.valid_indices = []  # 存放每一折的valid_index

    def get_transformer(self):
        transformer = Pipeline([
            ('same_house_transformer', e_tool.SameHouseTransformer()),
            ('similar_house_transformer', e_tool.SimilarHouseTransformer()),
            ('same_building_transformer', e_tool.PossibeSameBuilding()),
            ('building_age_local_transformer_1',
             e_tool.DistMeanTransformer(threshold=500,
                                        groupby_name='area',
                                        target_name='building_age')),
            ('building_age_local_transformer_2',
             e_tool.DistMeanTransformer(threshold=1000,
                                        groupby_name='area',
                                        target_name='building_age')),
            ('dist_transformer_4',
             e_tool.DistMeanTransformer(threshold=500, groupby_name='area')),
            ('dist_transformer_3',
             e_tool.DistMeanTransformer(threshold=1000,
                                        groupby_name='area')),  # noqa
            ("post_preprocess_transformer",
             e_tool.PostPreprocessTransformer(threshold=500)),
            ('raw_transformer',
             e_tool.RawFeatExtracter(
                 numeric=e_config.RAW_CONFIG['numeric'],
                 cat=e_config.RAW_CONFIG['cat'],
                 target_mean=e_config.RAW_CONFIG['target_mean'],
                 target_count=e_config.RAW_CONFIG['target_count'],
                 target_var=e_config.RAW_CONFIG['target_var'],
                 remove_cols=e_config.RAW_CONFIG['remove_cols'])),
        ])
        return transformer

    def set_config(self):
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'custom',
            'boosting_type': 'gbdt',
            'seed': 16,
            'learning_rate': 0.01,
            'num_leaves': 162,
            'max_depth': 37,
            'verbosity': -1,
            'n_jobs': -1,
            'lambda_l1': 9.191748589273263e-05,
            'lambda_l2': 3.3935617389132826e-08,
            'feature_fraction': 0.31448447806937796,
            'bagging_fraction': 0.9922651680171057,
            'bagging_freq': 3,
            'min_data_in_leaf': 14
        }

    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.DataFrame,
              selected_feats=None):
        self.models = []
        self.mape = []
        for fold_n, (train_index, valid_index) in enumerate(
                self.fold_handler.split(X_train, y_train)):
            print('Fold {}'.format(fold_n + 1))
            X_train_fold, X_valid_fold = X_train.iloc[train_index].copy(
            ), X_train.iloc[valid_index].copy()
            y_train_fold, y_valid_fold = y_train.iloc[train_index].copy(
            ), y_train.iloc[valid_index].copy()
            transformer = self.get_transformer()
            X_train_fold = transformer.fit_transform(X_train_fold,
                                                     y_train_fold)
            X_valid_fold = transformer.transform(X_valid_fold)
            if selected_feats is not None:
                X_train_fold = X_train_fold[selected_feats]
                X_valid_fold = X_valid_fold[selected_feats]
            train_data = lgb.Dataset(
                X_train_fold,
                label=y_train_fold,
                categorical_feature=e_config.RAW_CONFIG['cat'],
            )
            valid_data = lgb.Dataset(
                X_valid_fold,
                label=y_valid_fold,
            )
            self.feature_names = X_train_fold.columns.tolist()
            callbacks = [
                lgb.early_stopping(stopping_rounds=500,
                                   first_metric_only=True),
                lgb.callback.log_evaluation(period=100)
            ]
            model = lgb.train(self.lgb_params,
                              train_data,
                              valid_sets=[train_data, valid_data],
                              num_boost_round=10000,
                              categorical_feature=e_config.RAW_CONFIG['cat'],
                              callbacks=callbacks,
                              fobj=lgb_mape,
                              feval=lgb_mape_eval)
            preds = model.predict(X_valid_fold)
            mape = np.mean(
                np.abs((y_valid_fold.values - preds.reshape(-1, 1)) /
                       y_valid_fold.values)) * 100
            self.mape.append(mape)
            self.models.append((transformer, model))
            # 保存預測結果和殘差
            self.predictions.append(preds)
            self.residuals.append(y_valid_fold.values - preds)
            self.valid_indices.append(valid_index)
            # # 畫預測值與實際值的散點圖
            # plt.scatter(y_valid_fold.values, preds)
            # plt.plot(
            #     [y_valid_fold.min(), y_valid_fold.max()],
            #     [y_valid_fold.min(), y_valid_fold.max()],
            #     'k--',
            #     lw=3)
            # plt.xlabel('Actual')
            # plt.ylabel('Predicted')
            # plt.title('Actual vs Predicted for Fold {}'.format(fold_n + 1))
            # plt.show()

            # self.plot_importances()

    def predict(self, X: pd.DataFrame, post_processing=True):
        if post_processing:
            pass
        preds = []
        for transformer, model in self.models:
            use_X = X.copy()
            use_X = transformer.transform(use_X)
            pred = model.predict(use_X)
            preds.append(pred)
        preds = np.mean(preds, axis=0)

        return preds

    def study_params(self, X_train, y_train, n_trials=20):

        def objective(trial):
            params = {
                'objective':
                'regression',
                'metric':
                'custom',
                'seed':
                16,
                'verbosity':
                -1,
                'boosting_type':
                'gbdt',
                'learning_rate':
                0.01,
                'n_jobs':
                -1,
                'lambda_l1':
                trial.suggest_float('lambda_l1', 1e-8, 15.0, log=True),
                'lambda_l2':
                trial.suggest_float('lambda_l2', 1e-8, 15.0, log=True),
                'max_depth':
                trial.suggest_int('max_depth', 5, 50),
                'num_leaves':
                trial.suggest_int('num_leaves', 30, 1024),
                'feature_fraction':
                trial.suggest_float('feature_fraction', 0.1, 1.0),
                'bagging_fraction':
                trial.suggest_float('bagging_fraction', 0.1, 1.0),
                'bagging_freq':
                trial.suggest_int('bagging_freq', 1, 20),
                'min_data_in_leaf':
                trial.suggest_int('min_data_in_leaf', 5, 50),
            }
            mape_values = []
            for fold_n, (train_index, valid_index) in enumerate(
                    self.fold_handler.split(X_train)):
                print('Fold {}'.format(fold_n + 1))
                # 使用 LOADER.load_train_data 方法加载数据
                X_train_fold, X_valid_fold = X_train.iloc[train_index].copy(
                ), X_train.iloc[valid_index].copy()
                y_train_fold, y_valid_fold = y_train.iloc[train_index].copy(
                ), y_train.iloc[valid_index].copy()
                try:
                    X_valid_fold = LOADER.load_train_data(
                        f'fold_{fold_n + 1}_X_valid_fold.joblib')
                    train_data = LOADER.load_train_data(
                        f'fold_{fold_n + 1}_train_data.joblib')
                    valid_data = LOADER.load_train_data(
                        f'fold_{fold_n + 1}_valid_data.joblib')
                except Exception:
                    transformer = self.get_transformer()
                    X_train_fold = transformer.fit_transform(
                        X_train_fold, y_train_fold)
                    X_valid_fold = transformer.transform(X_valid_fold)
                    train_data = lgb.Dataset(
                        X_train_fold,
                        label=y_train_fold,
                        categorical_feature=e_config.RAW_CONFIG['cat'],
                    )
                    valid_data = lgb.Dataset(
                        X_valid_fold,
                        label=y_valid_fold,
                    )
                callbacks = [
                    lgb.early_stopping(stopping_rounds=500,
                                       first_metric_only=True),
                    lgb.callback.log_evaluation(period=100)
                ]
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[train_data, valid_data],
                    num_boost_round=10000,
                    categorical_feature=e_config.RAW_CONFIG['cat'],
                    callbacks=callbacks,
                    fobj=lgb_mape,
                    feval=lgb_mape_eval)
                preds = model.predict(X_valid_fold)
                mape = np.mean(
                    np.abs((y_valid_fold.values - preds.reshape(-1, 1)) /
                           y_valid_fold.values)) * 100
                mape_values.append(mape)
            return np.mean(mape_values, axis=0)

        study = optuna.create_study(direction='minimize')
        study.enqueue_trial(self.lgb_params)
        study.optimize(objective, n_trials=n_trials)
        # 輸出最佳參數
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial
        print('Value: ', trial.value)
        print('Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
            self.lgb_params[key] = value

    def feature_selection(self,
                          X_train,
                          y_train,
                          importance_type='gain',
                          threshold=None):
        # 先使用所有特征进行训练，获取特征重要性
        transformer = self.get_transformer()
        X_train = transformer.fit_transform(X_train, y_train)

        # 先使用所有特征进行训练，获取特征重要性
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=e_config.RAW_CONFIG['cat'],
            free_raw_data=False)

        model = lgb.train(self.lgb_params,
                          train_data,
                          num_boost_round=100,
                          fobj=lgb_mape,
                          feval=lgb_mape_eval)

        # 使用模型的 feature_importance() 方法获取重要性
        feature_importances = model.feature_importance(
            importance_type=importance_type)

        # 如果未指定阈值，则使用特征重要性的中位数作为阈值
        if threshold is None or threshold == 'median':
            threshold = np.median(feature_importances)

        # 根据特征重要性筛选特征
        important_features_indices = np.where(
            feature_importances >= threshold)[0]
        important_features = [
            X_train.columns[i] for i in important_features_indices
        ]

        # 更新模型使用的特征
        self.selected_feature_names = important_features

        # 选择重要的特征并返回相应的 DataFrame
        return X_train.loc[:, important_features]

    def plot_importances(self):
        importances = self.models[-1][-1].feature_importance(
            importance_type='gain')
        # 將特徵重要性與特徵名稱對應起來
        feature_importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })

        # 將特徵重要性排序
        feature_importances = feature_importances.sort_values('importance',
                                                              ascending=False)

        # 繪製特徵重要性
        plt.figure(figsize=(10, 6))
        plt.title("Feature importances")
        plt.barh(feature_importances['feature'].iloc[:40],
                 feature_importances['importance'].iloc[:40],
                 align='center')
        plt.gca().invert_yaxis()
        plt.show()
