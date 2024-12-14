import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """Load and Merge the arguments datasets.

    Args:
    messages_filepath: csv file. contains the actual messages sent during disaster events. 
    categories_filepath: csv file. contains the categories associated with each message.

    Returns:
    df: pandas one dataFrame acheived by merging the args.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id') 
    return df

def clean_data(df):
    """Clean the Dataframe that has been merged 

    Args: 
    df: the dataframe that contains the messages and categories

    returns: 
    dataframe after cleaning process
    """

    # Split the Categories
    categories = df['categories'].str.split(';', expand=True) 
    # Create a list of new column names based on the first row of the categories DataFrame
    category_colnames = [category.split('-')[0] for category in categories.iloc[0]]

    # Rename the columns of the categories DataFrame
    categories.columns = category_colnames

    # Convert the values to integers and handle the 'related' column specifically
    for column in categories:
        # Ensure all values are strings before applying .str
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    # Validate 'related' column values before cleaning
    # print("Before cleaning, 'related' column value counts:")
    # print(categories['related'].value_counts())

    # Handle the 'related' column specifically to drop rows with value 2
    if 'related' in categories.columns:
        categories = categories[categories['related'] != 2]


    # Validate 'related' column values after cleaning
    # print("After cleaning, 'related' column value counts:")
    # print(categories['related'].value_counts())

    # Drop the original 'categories' column from the main DataFrame
    df = df.drop('categories', axis=1)

    # Concatenate the original DataFrame with the new categories DataFrame
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
        """Save df in the SQL server

        Args:
        df: the dataframe to be saved
        database_filename: the file name of the saved data 
        """
        engine = create_engine('sqlite:///'+database_filename)
        df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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