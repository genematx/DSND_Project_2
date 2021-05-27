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


class LengthEstimator(BaseEstimator, TransformerMixin):
    """A custom transformer class to estimate the length of a message."""

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **fit_params):
        return pd.DataFrame(pd.Series(X).str.len())


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
        cat_names: list of str
            List of categories names.
    """

    engine = create_engine('sqlite:///'+db_filepath)
    df = pd.read_sql_table('merged', engine).sample(frac=sample_frac, random_state=42)
    X = df['message']
    Y = df[ [c for c in df.columns if c not in ('id', 'message', 'original', 'genre', 'child_alone')] ]
    cat_names = Y.columns

    return X, Y, cat_names


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
    """Define and train the classification pipeline.

        The pipeline takes as its input a string representing a text message,
        transforms it into TF-IDF vector, and runs a multi-target classification
        algorithm with 36 targets.

    """

    # A (sub-)pipeline to vectorize the message and ombine it with a text-length feature
    pl_NLP = FeatureUnion([
                    ('pl_Txt', Pipeline([
                                        ('vectr', CountVectorizer(tokenizer=tokenize, max_features=1500)),
                                        ('tfidf', TfidfTransformer()),
                                ]) ),
                    ('length', LengthEstimator())
                ])

    # Add an SVC classifier
    pl_Final = Pipeline([
                    ('pl_NLP', pl_NLP),
                    ('clsfr', MultiOutputClassifier(SVC(gamma='auto')))
               ])

    # Run GridSearch to optimize some of the parameters
    parameters = {'pl_NLP__pl_Txt__vectr__max_df' : [1.0, 0.9],
              'clsfr__estimator__kernel' : ['rbf', 'poly']}

    cv = GridSearchCV(pl_Final, param_grid=parameters, cv=3, verbose=4, scoring='f1_weighted')

    return cv


def evaluate_model(model, X_test, Y_test, cat_names):
    """Evaluate the trained model and display classification reports.
        Args:
            X_test, Y_test : pd.DataFrame or np.array
                Testing sets of predictors (messages) and labels.
            cat_names: list of str
                Names of the classification categories.
        Returns:
            Y_pred: pd.DataFrame
                Predicted labels.
    """

    Y_pred = pd.DataFrame(model.predict(X_test), columns = cat_names)

    for i, cat in enumerate(cat_names):
        print('Classification report for category #{:d} - {}'.format(i, cat))
        print(classification_report(Y_test[cat], Y_pred[cat]))

    return Y_pred


def save_model(model, model_filepath):
    """Save the trained model in a pkl file."""
    with open(model_filepath, 'wb') as fp:
        dill.dump(model, fp)


def main():
    if len(sys.argv) == 3:
        db_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(db_filepath))
        X, Y, cat_names = load_data(db_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, cat_names)

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
