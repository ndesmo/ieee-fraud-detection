import pandas as pd
import numpy as np
import sys
import os
import re

import pickle

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn.under_sampling import RandomUnderSampler

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from sklearn import metrics

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from ieee_fraud_detection.data.process_data import save_data
from ieee_fraud_detection.models.selection_functions import *

def load_data(database_filepath):
    """
    Load the preprocessed data from the database file and split it into
    X, Y and category names datasets.

    :param database_filepath: Filepath of the database file
    :return: Pandas dataframes for the X, Y and category names
    """

    # Set up the SQL Alchemy engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql('transactions', engine)

    X = df[df.columns[1:]]
    y = df.isFraud

    return X, y


col_dict = {
    'win': 'Windows',
    'mac': 'Mac',
    'samsung': 'Samsung',
    'sm-': 'Samsung mobile',
    'trident': 'Trident',
    'huawei': 'Huawei',
    'ios': 'iOS',
    'lg': 'LG',
    'moto': 'Moto',
    'rv': 'rv',
    'redmi': 'Redmi',
    'htc': 'HTC',
    'gt-': 'GT',
    'hisense': 'Hisense',
    'blade': 'Blade',
    'alcatel': 'Alcatel',
    'linux': 'Linux',
    'nexus': 'Nexus',
    'asus': 'Asus'
}


def regex_replace(col_dict, text):
    regex = re.compile("(%s)" % "|".join(
        map(re.escape, col_dict.keys())
    ), re.IGNORECASE)
    if regex.search(text):
        ret = regex.search(text)
        return col_dict[ret.group().lower()]
    else:
        return 'Other'


def preprocess_data(X, y):
    """
    Extra preprocessing of data
    :param X: Features dataset
    :param y: Response dataset
    :return: Further preprocessed dataset
    """
    
    # Encode time based variables
    X['mins'] = np.floor(X.TransactionDT / 60)
    X['hours'] = np.floor(X['mins'] / 60)
    X['days'] = np.floor(X['hours'] / 24)
    X['weeks'] = np.floor(X['days'] / 7)
    X['months'] = np.floor(X['days'] / 365 * 12)
    X['years'] = np.floor(X['days'] / 365)
    
    X['dayofweek'] = np.mod(X['days'] - X['weeks'] * 7, 7)
    X['dayofmonth'] = np.floor(np.mod(X['days'] - np.floor(X['months'] * 365 / 12), 365 / 12))
    
    X['hourofday'] = np.mod(X['hours'] - X['days'] * 24, 24)

    X['DevicePlatform'] = X['DeviceInfo'].apply(lambda v: regex_replace(col_dict, str(v)))
    
    fillna_cols = ['card4', 'card5', 'card6',
                   'P_emaildomain', 'R_emaildomain', 'M4',
                   'dist1', 'dist2',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                   'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                   'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31',
                   'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41',
                   'V42', 'V43', 'V44', 'V45', 'V46',
                   'DeviceType', 'DeviceInfo',
                   'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_34'
                   ]
    X[fillna_cols] = X[fillna_cols].fillna('NaN')
    
    return X, y


def resample_data(X, y, cols=None):
    """
    Preprocess and resample the data to balance out the classes
    :param X: Features dataset
    :param y: Response dataset
    :return: Resampled X and y
    """
    
    if cols is None:
        cols = X.columns

    rus = RandomUnderSampler()
    X_rs, y_rs = rus.fit_sample(X[cols], y)

    return X_rs, y_rs


def import_data(database_filepath, cols, resample=True, resample_suffix='_rs'):
    """
    Loads data, optionally resampling data and caching locally
    :param database_filepath:  Filepath of the database file
    :param resample: Boolean indicating whether to resample data if not already cached locally
    :param resample_suffix: filename suffix of the resampled dataset to store locally
    :return: 
    """
    
    if resample:
        
        rs_dir, rs_path = os.path.split(database_filepath)
        rs_fn, rs_ext = os.path.splitext(rs_path)
        rs_fp = os.path.join(rs_dir, rs_fn + resample_suffix + rs_ext)
        
        # Use previously created resampled data
        if os.path.exists(rs_fp):
            
            X_rs, y_rs = load_data(rs_fp)

            return X_rs, y_rs
        
        else:

            # Load and resample
            X, y = load_data(database_filepath)
            X, y = preprocess_data(X, y)
            X_rs, y_rs = resample_data(X, y, cols)

            # Combine
            X_rs = pd.DataFrame(X_rs, columns=cols)
            y_rs = pd.DataFrame(pd.Series(y_rs), columns=['isFraud'])
            df_rs = pd.concat([y_rs, X_rs], axis=1)

            # Save locally
            save_data(df_rs, rs_fp)

            return X_rs, y_rs

    # Not performing resampling
    else:

        X, y = load_data(database_filepath)

        return X, y


class ModifiedLabelEncoder(LabelEncoder):

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)


def build_model():
    """
    Initialize a model for running the data through
    :return: a model
    """
    
    onehot = ('onehot_encoder', OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))

    # Set up the pipeline

    ## NUMERICALS
    pipe_TransactionAmt = Pipeline([
        ('column_selection', FunctionTransformer(col_TransactionAmt, validate=False))
    ])
    pipe_V1 = Pipeline([
        ('column_selection', FunctionTransformer(col_V1, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V1NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V1NaN, validate=False))
    ])
    pipe_V2 = Pipeline([
        ('column_selection', FunctionTransformer(col_V2, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V2NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V2NaN, validate=False))
    ])
    pipe_V3 = Pipeline([
        ('column_selection', FunctionTransformer(col_V3, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V3NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V3NaN, validate=False))
    ])
    pipe_V4 = Pipeline([
        ('column_selection', FunctionTransformer(col_V4, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V4NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V4NaN, validate=False))
    ])
    pipe_V5 = Pipeline([
        ('column_selection', FunctionTransformer(col_V5, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V5NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V5NaN, validate=False))
    ])
    pipe_V6 = Pipeline([
        ('column_selection', FunctionTransformer(col_V6, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V6NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V6NaN, validate=False))
    ])
    pipe_V7 = Pipeline([
        ('column_selection', FunctionTransformer(col_V7, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V7NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V7NaN, validate=False))
    ])
    pipe_V8 = Pipeline([
        ('column_selection', FunctionTransformer(col_V8, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V8NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V8NaN, validate=False))
    ])
    pipe_V9 = Pipeline([
        ('column_selection', FunctionTransformer(col_V9, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V9NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V9NaN, validate=False))
    ])
    pipe_V10 = Pipeline([
        ('column_selection', FunctionTransformer(col_V10, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V10NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V10NaN, validate=False))
    ])
    pipe_V11 = Pipeline([
        ('column_selection', FunctionTransformer(col_V11, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V11NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V11NaN, validate=False))
    ])
    pipe_V12 = Pipeline([
        ('column_selection', FunctionTransformer(col_V12, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V12NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V12NaN, validate=False))
    ])
    pipe_V13 = Pipeline([
        ('column_selection', FunctionTransformer(col_V13, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V13NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V13NaN, validate=False))
    ])
    pipe_V14 = Pipeline([
        ('column_selection', FunctionTransformer(col_V14, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V14NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V14NaN, validate=False))
    ])
    pipe_V15 = Pipeline([
        ('column_selection', FunctionTransformer(col_V15, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V15NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V15NaN, validate=False))
    ])
    pipe_V16 = Pipeline([
        ('column_selection', FunctionTransformer(col_V16, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V16NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V16NaN, validate=False))
    ])
    pipe_V17 = Pipeline([
        ('column_selection', FunctionTransformer(col_V17, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V17NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V17NaN, validate=False))
    ])
    pipe_V18 = Pipeline([
        ('column_selection', FunctionTransformer(col_V18, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V18NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V18NaN, validate=False))
    ])
    pipe_V19 = Pipeline([
        ('column_selection', FunctionTransformer(col_V19, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='most_frequent'))
    ])
    pipe_V19NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_V19NaN, validate=False))
    ])


    pipe_dist1 = Pipeline([
        ('column_selection', FunctionTransformer(col_dist1, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='constant', fill_value=0))
    ])
    pipe_dist1NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_dist1NaN, validate=False))
    ])
    pipe_dist2 = Pipeline([
        ('column_selection', FunctionTransformer(col_dist2, validate=False)),
        ('imputer', SimpleImputer(missing_values='NaN', strategy='constant', fill_value=0))
    ])
    pipe_dist2NaN = Pipeline([
        ('column_selection', FunctionTransformer(col_dist2NaN, validate=False))
    ])
    
    
    ## CATEGORICALS
    pipe_ProductCD = Pipeline([
        ('column_selection', FunctionTransformer(col_ProductCD, validate=False)),
        onehot
    ])
    pipe_card4 = Pipeline([
        ('column_selection', FunctionTransformer(col_card4, validate=False)),
        onehot
    ])
    pipe_card5 = Pipeline([
        ('column_selection', FunctionTransformer(col_card5, validate=False)),
        onehot
    ])
    pipe_card6 = Pipeline([
        ('column_selection', FunctionTransformer(col_card6, validate=False)),
        onehot
    ])
    pipe_P_emaildomain = Pipeline([
        ('column_selection', FunctionTransformer(col_P_emaildomain, validate=False)),
        onehot
    ])
    pipe_R_emaildomain = Pipeline([
        ('column_selection', FunctionTransformer(col_R_emaildomain, validate=False)),
        onehot
    ])
    pipe_M4 = Pipeline([
        ('column_selection', FunctionTransformer(col_M4, validate=False)),
        onehot
    ])
    pipe_DeviceType = Pipeline([
        ('column_selection', FunctionTransformer(col_DeviceType, validate=False)),
        onehot
    ])
    pipe_DevicePlatform = Pipeline([
        ('column_selection', FunctionTransformer(col_DevicePlatform, validate=False)),
        onehot
    ])
    pipe_id_12 = Pipeline([
        ('column_selection', FunctionTransformer(col_id_12, validate=False)),
        onehot
    ])
    pipe_id_15 = Pipeline([
        ('column_selection', FunctionTransformer(col_id_15, validate=False)),
        onehot
    ])
    pipe_id_16 = Pipeline([
        ('column_selection', FunctionTransformer(col_id_16, validate=False)),
        onehot
    ])
    pipe_id_23 = Pipeline([
        ('column_selection', FunctionTransformer(col_id_23, validate=False)),
        onehot
    ])
    pipe_id_27 = Pipeline([
        ('column_selection', FunctionTransformer(col_id_27, validate=False)),
        onehot
    ])
    pipe_id_28 = Pipeline([
        ('column_selection', FunctionTransformer(col_id_28, validate=False)),
        onehot
    ])
    pipe_id_29 = Pipeline([
        ('column_selection', FunctionTransformer(col_id_29, validate=False)),
        onehot
    ])
    pipe_id_34 = Pipeline([
        ('column_selection', FunctionTransformer(col_id_34, validate=False)),
        onehot
    ])
    pipe_hourofday = Pipeline([
        ('column_selection', FunctionTransformer(col_hourofday, validate=False)),
        onehot
    ])
    pipe_dayofweek = Pipeline([
        ('column_selection', FunctionTransformer(col_dayofweek, validate=False)),
        onehot
    ])
    pipe_dayofmonth = Pipeline([
        ('column_selection', FunctionTransformer(col_dayofmonth, validate=False)),
        onehot
    ])

    pipe = Pipeline([
        ('union', FeatureUnion([
            ('pipe_TransactionAmt', pipe_TransactionAmt),
            ('pipe_V1', pipe_V1),
            ('pipe_V1NaN', pipe_V1NaN),
            ('pipe_V2', pipe_V2),
            ('pipe_V2NaN', pipe_V2NaN),
            ('pipe_V3', pipe_V3),
            ('pipe_V3NaN', pipe_V3NaN),
            ('pipe_V4', pipe_V4),
            ('pipe_V4NaN', pipe_V4NaN),
            ('pipe_V5', pipe_V5),
            ('pipe_V5NaN', pipe_V5NaN),
            ('pipe_V6', pipe_V6),
            ('pipe_V6NaN', pipe_V6NaN),
            ('pipe_V7', pipe_V7),
            ('pipe_V7NaN', pipe_V7NaN),
            ('pipe_V8', pipe_V8),
            ('pipe_V8NaN', pipe_V8NaN),
            ('pipe_V9', pipe_V9),
            ('pipe_V9NaN', pipe_V9NaN),
            ('pipe_V10', pipe_V10),
            ('pipe_V10NaN', pipe_V10NaN),
            ('pipe_V11', pipe_V11),
            ('pipe_V11NaN', pipe_V11NaN),
            ('pipe_V12', pipe_V12),
            ('pipe_V12NaN', pipe_V12NaN),
            ('pipe_V13', pipe_V13),
            ('pipe_V13NaN', pipe_V13NaN),
            ('pipe_V14', pipe_V14),
            ('pipe_V14NaN', pipe_V14NaN),
            ('pipe_V15', pipe_V15),
            ('pipe_V15NaN', pipe_V15NaN),
            ('pipe_V16', pipe_V16),
            ('pipe_V16NaN', pipe_V16NaN),
            ('pipe_V17', pipe_V17),
            ('pipe_V17NaN', pipe_V17NaN),
            ('pipe_V18', pipe_V18),
            ('pipe_V18NaN', pipe_V18NaN),
            ('pipe_V19', pipe_V19),
            ('pipe_V19NaN', pipe_V19NaN),
            ('pipe_dist1', pipe_dist1),
            ('pipe_dist1NaN', pipe_dist1NaN),
            ('pipe_dist2', pipe_dist2),
            ('pipe_dist2NaN', pipe_dist2NaN),
            
            
            ('pipe_ProductCD', pipe_ProductCD),
            ('pipe_card4', pipe_card4),
            # ('pipe_card5', pipe_card5),
            ('pipe_card6', pipe_card6),
            ('pipe_P_emaildomain', pipe_P_emaildomain),
            ('pipe_R_emaildomain', pipe_R_emaildomain),
            ('pipe_M4', pipe_M4),
            ('pipe_DeviceType', pipe_DeviceType),
            ('pipe_DevicePlatform', pipe_DevicePlatform),
            ('pipe_id_12', pipe_id_12),
            ('pipe_id_15', pipe_id_15),
            ('pipe_id_16', pipe_id_16),
            ('pipe_id_23', pipe_id_23),
            ('pipe_id_27', pipe_id_27),
            ('pipe_id_28', pipe_id_28),
            ('pipe_id_29', pipe_id_29),
            # ('pipe_id_34', pipe_id_34),

            ('pipe_hourofday', pipe_hourofday),
            ('pipe_dayofweek', pipe_dayofweek),
            ('pipe_dayofmonth', pipe_dayofmonth)
        ])),
        # ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=10))),
        ('clf', RandomForestClassifier())
    ])

    # Set up a parameter grid
    pg = [
        {
            'clf': [RandomForestClassifier()],
            'clf__n_estimators': [10],
            'clf__criterion': ['entropy'], # ['gini', 'entropy'],
            'clf__min_samples_split': [10], # [10,20,50,100],
            'clf__min_samples_leaf': [1], # [1,2,5,10],
            'clf__min_weight_fraction_leaf': [0], # [0, 0.01, 0.1, 0.5],
            'clf__max_features': [None], # ['auto', 'sqrt', 'log2', None],
            'clf__bootstrap': [True], # [True, False],
            'clf__oob_score': [False], # [False, True],
            # 'clf__random_state': [0],
            # 'clf__ccp_alpha': [0.0, 0.05, 0.10, 0.5, 1.0]
        }
    ]

    return GridSearchCV(
        pipe, param_grid=pg, cv=10
    )


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model results by printing the model scores for each category.
    :param model:
    :param X_test: The vectorized text features of the test data.
    :param Y_test: The response variables of the test data
    :return:
    """
    # Predict the output of the model on the test data
    y_pred = model.predict(X_test)

    # Print the accuracy score
    print('Accuracy: {}\nPrecision: {}\nRecall: {}\n==============\n'.format(
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro')
    ))
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

    print('Overall evaluation:')

    # Output the GridSearchCV best score and best params
    print('The best score from GridSearchCV: {}'.format(model.best_score_))
    print('Best model parameters:')
    print(model.best_params_)


def save_model(model, model_filepath):
    """
    Save the model as a pickle file to the filepath given.
    :param model: The model used to fit the data
    :param model_filepath: The filepath to save the model to
    :return:
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main(database_filepath='../data/train_transactions.db',
         model_filepath='clf_transactions.pkl'):
    """
    Processing function for the full sequence. Loads the data from the database,
    splits into train and test datasets, runs the model pipeline and fits the data
    to the model. Evaluation of the model is then performed and the model is output
    to a pickle file.

    :param database_filepath: The filepath of the database file to read.
    :param model_filepath: The filepath to save the model to.
    :return:
    """

    # Optional override with command line
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
    
    print('Importing data...')

    X, y = import_data(
        database_filepath, 
        ['TransactionAmt', 'ProductCD',
           'card4', 'card5', 'card6',
           'P_emaildomain', 'R_emaildomain', 'M4',
           'dist1', 'dist2',
           'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
           'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
           'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31',
           'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41',
           'V42', 'V43', 'V44', 'V45', 'V46',
           'DeviceType', 'DevicePlatform',
           'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_34',
           'weeks', 'months', 'hourofday', 'dayofweek', 'dayofmonth'
        ]
    )

    print(X.shape)

    print('Creating train test split from data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, y_test)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()