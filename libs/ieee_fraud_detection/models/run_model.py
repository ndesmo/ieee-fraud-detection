from ieee_fraud_detection.models.train_classifier import preprocess_data
from ieee_fraud_detection.data.process_data import load_data

from sklearn.externals import joblib

import pandas as pd
import sys

transaction_filepath = sys.argv[1]
identity_filepath = sys.argv[2]
model_filepath = sys.argv[3]
output_data = sys.argv[4]

# load data
print('Loading data...')
X = load_data(transaction_filepath, identity_filepath)

print(X.head)

print(X.columns)

# preprocess
print('Preprocessing data...')
X = preprocess_data(X)

# load model
print('Loading model...')
model = joblib.load(model_filepath)

# run predictions
print('Running predictions...')
y_pred = model.predict(X)

# create dataframe
print('Creating dataframe...')
df_preds = pd.DataFrame({
    'TransactionID': X.index.values,
    'isFraud': y_pred
})

# save csv
print('Saving to CSV file...')
df_preds.to_csv(output_data, index=False)

print('Done!')