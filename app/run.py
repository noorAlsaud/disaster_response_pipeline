import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from sqlalchemy import create_engine
import joblib
from sqlalchemy import create_engine
from plotly.graph_objs import Heatmap
app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

database_filename = 'data/DisasterResponse.db'

# load data
engine = create_engine('sqlite:///'+database_filename)
df = pd.read_sql_table('DisasterResponse', engine)
# load model            
model = joblib.load("models/classifier.pkl")


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
                Bar(
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
        
    ] 

    """
    Visual (1): Distribution of Message Categories
    this data needed to represents the Distribution of Massage Categories as a Bar chart, 
    X-Axis: Categories of messages 
    Y-Axis: total number of messages thaat lay into each category 
    
    Higher bars indicate categories with more messages. 

    """
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    graphs.append({
    'data': [
        Bar(
            x=category_names,
            y=category_counts
        )
    ],
    'layout': {
        'title': 'Distribution of Message Categories',
        'yaxis': {'title': "Count"},
        'xaxis': {'title': "Category", 'tickangle': -45}
    }
})
   
    
    """
    Visual (2): Correlation Heatmap of Message Categories
    This data is used to represent the Correlation Heatmap of Message Categories, 
    showing the relationship between different categories of messages.

    X-Axis: Categories of messages
    Y-Axis: Categories of messages

    The color intensity indicates the strength of the correlation between categories:
    - Darker colors represent higher correlations.
    - Lighter colors represent lower correlations.

    This visualization helps to identify how often different categories co-occur.
    """
    category_corr = df.iloc[:, 4:].corr()
    graphs.append({
    'data': [
        Heatmap(
            z=category_corr.values,
            x=category_corr.columns,
            y=category_corr.columns,
            colorscale='Viridis'
        )
    ],
    'layout': {
        'title': 'Correlation Heatmap of Message Categories',
        'xaxis': {'title': "Category"},
        'yaxis': {'title': "Category"}
    }
})
   
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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

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