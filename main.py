import utils_land
import dataloader
import etl_tool as e_tool
import pandas as pd
import numpy as np
import trainer

# 產生other_prices.joblib
print("Processing other_prices.joblib...")
other_prices = utils_land.processing_land()
print("Processing Done!")
print(other_prices.head())

# preprocessing the data
loader = dataloader.DataLoader()
print("Processing training_data.csv")
train = loader.load_train_data('training_data.csv', data_type='csv')
train = e_tool.preprocess_raw_data(train)
print("train data processing Done!")
X_train = train.drop(['y'], axis=1).copy()
y_train = train[['y']].copy()

# training processing
print("Initital the trainer")
lgb_trainer = trainer.LGBTrainer()
print("start training")
lgb_trainer.train(X_train, y_train)

# merge the public and test
print("Merge the public and test")
public = loader.load_train_data('public_dataset.csv', data_type='csv')
private = loader.load_train_data('private_dataset.csv', data_type='csv')
test = pd.concat([public, private]).reset_index(drop=True)
print("Start preprocessing the test data")
test = e_tool.preprocess_raw_data(test)

# prediction
print("test data processing Done!")
print("Start internal data preprocessing and prediction")
all_preds = lgb_trainer.predict(test)
print("internal data prediction Done!")

# get the final prediction
post_same_transformer = e_tool.PostSameHouseTransformer().fit(X_train, y_train)
step_transformer = e_tool.PostStepTransformer()
a_sample = loader.load_train_data('public_dataset.csv', data_type='csv')
a_private = loader.load_train_data('private_dataset.csv', data_type='csv')
a_sample = pd.concat([a_sample, a_private]).reset_index(drop=True)
a_sample = a_sample[['ID']]
a_sample["predicted_price"] = all_preds
samples_trans = a_sample.copy()
samples_trans.columns = ['id', 'y']
sample_2 = post_same_transformer.transform(samples_trans)
sample_2.columns = ['ID', 'predicted_price']
all_preds = a_sample['predicted_price'].values

sample_2['predicted_price'] = np.array(
    all_preds) * 0.7 + sample_2['predicted_price'] * 0.3
sample_2['predicted_price'] = step_transformer.transform(
    sample_2['predicted_price'])
sample_2.to_csv(loader.file_path/'submission.csv', index=None)
