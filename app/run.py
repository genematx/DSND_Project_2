import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Sunburst
from sklearn.externals import joblib
from sqlalchemy import create_engine
import dill
import pip

import re
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

app = Flask(__name__)

def install_with_pip(package, args=None):
    """Insatlls a package with pip.
    Args:
        package: str
            Name of the package to install
        args: list of str
            List of command attributes, e.g. ['-U', '--force-reinstall']
    """

    if args is None:
        args = []

    if hasattr(pip, 'main'):
        pip.main(['install', package]+args)
    else:
        pip._internal.main(['install', package]+args)

def classify(model, query):
    """Classify a query using a trained model.
    Args:
        model: sklearn.pipeline.Pipeline
            A pretrained classification model.
        query: str
            A message that needs to be classified.
    Returns:
        cls_labels: list of int
            Results of classification with each entry correponding to a specific
            class in the order of columns in the original database.
    """

    print('Classifying message \'{}\' ...'.format(query))

    return model.predict([query]).ravel()

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('merged', engine)
classification_labels = df.columns[4:]

# load model
with open("../models/classifier.pkl", 'rb') as fp:
        model = dill.load(fp)

classify(model, 'Query to classify')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Count related and unrelated messages
    related_counts = df[['related', 'request', 'offer']].sum()
    related_counts['unrelated'] = len(df)-related_counts['related']
    related_counts['related'] -= (related_counts.request+related_counts.offer)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Sunburst(
                            labels=["Total", "Related", "Unrelated", "Request", "Offer"],
                            parents=["", "Total", "Total", "Related", "Related"],
                            values=[0, related_counts.related, related_counts.unrelated,
                                    related_counts.request, related_counts.offer],
                            hoverinfo='label+percent parent'
                        )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                ),
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'grid': {'rows': 1, 'columns': 2, 'pattern': 'independent'},
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    print(query)

    # use model to predict classification for query
    classification_result = {label:value for label, value in \
                            zip(classification_labels, classify(model, query))}

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result = classification_result
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    # Update plotly to display Sunburst charts
    print(int(plotly.__version__[0]) == 4)

    main()
