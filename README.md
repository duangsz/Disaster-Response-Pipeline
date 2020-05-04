# Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Results
5. Licensing, Authors, and Acknowledgements
6. Instruction

# Installation
Besides Anaconda3, the punkt, stopwords and wordnet packages under nltk need to be installed

# Project Motivation
For this project, I was building ETL and machine learning pipelines to analyze message people sent out during disasters. The outcome will be integrated with an API that classifies disaster messages.

# File Descriptions
There are three folders:
1. data
* disaster_categories.csv: raw data for message categories
* disaster_messages.csv: raw data for message content
* process_data.py: ETL pipeline for reading in raw data and output cleaned data into database

2. models
* train_classifier.py: the machine learning pipeline to train and export model

3. app
* run.py: file for running the application
* templates: html files for the application

# Results
1. ETL pipeline output cleaned message and category data
2. Machine learning pipeline output the trained model that would classify disaster message on 36 categories
3. The web application would show three disaster message categories related visualizations and show the categories for each message user entered

# Licensing, Authors, Acknowledgements
You can find the Licensing for the data and other descriptive information at the Kaggle link available [here]( https://www.kaggle.com/airbnb/seattle/data#listings.csv). 

# Instruction
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
