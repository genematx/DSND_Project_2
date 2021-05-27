import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as go
from sklearn.externals import joblib
from sqlalchemy import create_engine
import dill


app = Flask(__name__)

def classify(model, query):
    """Classify a query using a trained model.
    Args:
        model: sklearn.pipeline.Pipeline
            A pretrained classification model.
        query: str
            A message that needs to be classified.
    Returns:
        cls_labels: dict
            Results of classification with keys corresponding to properties (derived
            from the column labels in the original df) and values -- to labels,
            e.g.: 'wethaer_related': 0.
    """
    cls_labels = {'related':1, 'offer':1, 'request':0, 'weather_related':1}
    return cls_labels

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('merged', engine)

# load model
with open("../models/classifier.pkl", 'rb') as fp:
        model = dill.load(fp)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                go.Bar(
                    x=genre_names,
                    y=genre_counts
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
    # cls_labels = model.predict([query])[0]
    classification_results = classify(model, query)

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
