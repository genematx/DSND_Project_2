import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load the datasets containing messages and categories.

    Args:
        messages_filepath: str
            Path to a .csv file with the messages dataset.
        categories_filepath: str
            Path to a .csv file with the corresponding categories.

    Returns:
        A combined dataframe.
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages, categories, on='id', how='outer')

    return df


def clean_data(df):
    """Cleans the megred dataframe of messages and categories."""

    # Remove duplicates
    df = df[~df.duplicated(subset='id')]

    # Split categories into 36 separate columns
    cat_split = df['categories'].str.split(';', expand=True)
    # Rename the categories columns; use the first row to infer the names
    cat_split.columns = cat_split.iloc[0,:].apply(lambda x : x[:-2])
    # Convert the entries to 0/1
    cat_split = cat_split.applymap(lambda x : x[-1]).astype('int')
    # Replace erroneous values ('2' corresponds to the 'No' answer)
    cat_split.replace({2:0}, inplace=True)
    # Combine with the rest of the dataframe
    df = df.loc[:, df.columns != 'categories'].join(cat_split)
    # Remove uninformative columns
    df = df.loc[:, df.columns != 'child_alone']

    return df


def save_data(df, database_filename):
    """Save the cleaned dataset to a SQLite database.

    Args:
        df: pandas.DataFrame
            Cleaned dataset; will be saved as a table called 'merged'.
        database_filename: str
            Filename of the database including the '.db' file extension.

    Returns:
        None
    """

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('merged', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
