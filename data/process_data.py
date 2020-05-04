import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    '''load message and category datasets, and merge them together based on ID column 
    to generate a full dataset
    
    INPUT: file paths for message and category datasets
    OUTPUT: full dataset in Pandas DataFrame format
    '''
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df=pd.merge(messages,categories,on=['id'])
    return df


def clean_data(df):
    '''Transform infomation in the category column
    drop duplicated rows
    
    INPUT: loaded dataframe
    OUTPUT: cleaned dataframe
    '''
    categories = df['categories'].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[0:(len(x)-2)])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.slice(-1)
        categories[column] = categories[column].astype(int)
    df.drop('categories',axis=1,inplace=True)
    df=pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''save cleaned dataframe to database'''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster', engine, index=False)  


def main():
    '''run the function to load data, clean data and upload data to database'''  
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