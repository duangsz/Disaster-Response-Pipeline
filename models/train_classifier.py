import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
lemmatizer=WordNetLemmatizer()
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''load data from the database, and extract varaiable X and Y
    INPUT: file path for database
    OUTPUT: Variable X, Y for classifier model building
    '''
    path='sqlite:///' + database_filepath
    engine = create_engine(path)
    df =pd.read_sql_table('Disaster',engine)
    X = df['message']
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    category_names =Y.columns
    return X,Y,category_names


def tokenize(text):
    '''clean and tokenize text, return cleaned tokenized text
    INPUT: Raw text
    OUTPUT: Cleaned tokenized text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detec_url=re.findall(url_regex,text)
    for url in detec_url:
        text=text.replace(url,"urlplaceholder")
    text_nopunc=re.sub(r'[^a-zA-Z0-9]'," ",text)
    text_nostop=[word for word in word_tokenize(text_nopunc.lower()) if word not in stopwords.words('english')]
    text_clean=[lemmatizer.lemmatize(word).strip() for word in text_nostop]
    return text_clean


def build_model():
    '''return Grid Search Model with pipeline and parameters
    '''
    pipeline= Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),('tfidf',TfidfTransformer()),('clf',MultiOutputClassifier(RandomForestClassifier()))])
    parameters={'clf__estimator__min_samples_split':[2, 5, 8]}
    cv=GridSearchCV(pipeline,parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate the model based on the average precision, recall and f1 score for all 36 categories
    INPUT: The grid search model, testing dataset for X and Y, and all 36 categories' names
    OUTPUT: Precision, recall and f1 score for all categories, and average values for the three metrics
    '''
    Y_pred=model.predict(X_test)
    Report={'Category':[],'Precision':[],'Recall':[],'f1':[]}
    num=0
    for col in category_names:
        print('start'+col+'1')
        precision,recall,fscore,support=precision_recall_fscore_support(Y_test[col],Y_pred[:,num],average='weighted')
        print('start'+col+'2')
        Report['Category'].append(col)
        Report['Precision'].append(precision)
        Report['Recall'].append(recall)
        Report['f1'].append(fscore)
        num=num+1
    print('start df_report')
    df_report=pd.DataFrame(Report)
    print('Average Precision : ',df_report['Precision'].mean())
    print('Average Recall : ', df_report['Recall'].mean())
    print('Average f1 score : ', df_report['f1'].mean())
    return df_report


def save_model(model, model_filepath):
    '''save model as a pickle file
    INPUT: model to be saved and path for the pickle file
    OUTPUT: a pickle file for the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()