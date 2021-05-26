import sys
import pandas as pd
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(db_filepath='DisasterResponse.db', sample_frac=1.0):
    """Load the data from database.
    Args:
        db_filepath: str
            Filepath to the SQL database.
        sample_frac: int
            Apply random subsampling and select only frac of entries.
    Returns:
        X, Y: np.ndarray or pandas.DataFrame
            Matrices of predictors (messages) and responses.
    """

    engine = create_engine('sqlite:///'+db_filepath)
    df = pd.read_sql_table('merged', engine).sample(frac=sample_frac, random_state=42)
    X = df['message']
    Y = df[ [c for c in df.columns if 'related' in c] ]

    return X, Y


def tokenize(text):
    """Turn a text string into an array of words.
        Args:
            text: str
        Returns:
            A list of lemmatised tokens (words).
    """

    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)

    text = nltk.word_tokenize(text)

    text = [w for w in text if w not in stopwords.words('english')]

    # Reduce words to their root form
    text = [WordNetLemmatizer().lemmatize(w, pos='v') for w in text]

    return text


def build_model():
    pipeline = Pipeline([
                ('vectr', CountVectorizer(tokenizer=tokenize, max_df=0.75, min_df=0.1)),
                ('tfidf', TfidfTransformer()),
#                 ('clf', RandomForestClassifier()),
                #         ('dense', DenseTransformer()),
#                 ('scale', StandardScaler()),
#                 ('clsfr', LinearSVC())
                ('clsfr', MultiOutputClassifier(LinearSVC()))
           ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    pass


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as fp:
        dill.dump(model, fp)


def main():
    if len(sys.argv) == 3:
        db_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(db_filepath))
        X, Y, category_names = load_data(db_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        # print('Evaluating model...')
        # evaluate_model(model, X_test, Y_test, category_names)

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
