import pandas as pd
import numpy as np
import sys
import os

import pickle

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn.under_sampling import RandomUnderSampler

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from ieee_fraud_detection.data.process_data import save_data

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

def col_TransactionAmt(X):
    return X[['TransactionAmt']]


def col_ProductCD(X):
    return X.ProductCD


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

    # Set up the pipeline
    pipe1 =  Pipeline([
        ('column_selection', FunctionTransformer(col_TransactionAmt, validate=False))
    ])
    pipe2 =  Pipeline([
        ('column_selection', FunctionTransformer(col_ProductCD, validate=False)),
        ('label_encoder', ModifiedLabelEncoder())
    ])

    pipe = Pipeline([
        ('union', FeatureUnion([
            ('pi1', pipe1),
            ('pi2', pipe2)
        ])),
        ('clf', GaussianNB())
    ])

    # Set up a parameter grid
    pg = [
        {
            'clf': [GaussianNB()]
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

    # print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    # X, y = load_data(database_filepath)
    # 
    # print('Resampling data...')
    # X, y = resample_data(X, y, cols=['TransactionAmt'])
    
    print('Importing data...')
    X, y = import_data(
        database_filepath, 
        ['TransactionAmt', 'ProductCD']
    )

    print('Creating train test split from data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(X_train.shape)
    print(y_train.shape)

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