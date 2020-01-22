import pandas as pd
import sys

from sqlalchemy import create_engine


def load_data(transaction_filepath, identity_filepath):
    """
    Loads the transaction and identity datasets from the local CSV files into
    pandas dataframes and merges them into a single pandas dataframe.

    :param transaction_filepath: The filepath for the transaction CSV file
    :param identity_filepath: The filepath for the identity CSV file
    :return: A merged pandas dataframe
    """

    # Load transaction dataset
    tr = pd.read_csv(transaction_filepath)

    # Load identity dataset
    id = pd.read_csv(identity_filepath)

    # Merge the datasets using a left join
    df = tr.merge(id, on='TransactionID', how='left', indicator=True)

    # Set the index to the transaction ID
    df = df.set_index('TransactionID')

    return df


def clean_data(df):
    """
    Applies transformations to the disaster dataset to prepare it for the ML code
    :param df: A pandas dataframe of disaster responses
    :return: A cleaned pandas dataframe
    """

    return df


def save_data(df, database_filename):
    """
    Saves the cleaned dataset into a local sqlite database file.

    :param df: The cleaned pandas dataframe
    :param database_filename: The filename of the database file
    :return:
    """

    # Set up the SQL Alchemy engine
    engine = create_engine('sqlite:///{}'.format(database_filename))

    # Save the pandas dataframe as a sqlite database file
    df.to_sql('transactions', engine, index=False)


def main(transaction_filepath='../input/train_transaction.csv',
         identity_filepath='../input/train_identity.csv',
         database_filepath='train_transactions.db'):

    # Allow command line override
    if len(sys.argv) == 4:
        transaction_filepath, identity_filepath, database_filepath = sys.argv[1:]

    print('Loading data...\n    TRANSACTION: {}\n    IDENTITY: {}'
          .format(transaction_filepath, identity_filepath))
    df = load_data(transaction_filepath, identity_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()