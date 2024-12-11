import pickle
import sys
import nltk
import pandas as pd
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    this method is to load the data from the SQLite 
    Args: 
    database_filepath : database file path in the project 

    returns: 
    X (pd.Series): A pandas Series containing the messages (feature data).
    Y (np.ndarray): A NumPy array containing the target labels for each category.
    category_names (list): A list of category names corresponding to the columns of Y.

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y.values, category_names


def tokenize(text):
    """
    this method is to Tokenize text 
     Args:
    text (str): The input text message to be tokenized and processed.

    Returns:
    clean_tokens: A list of cleaned tokens (words) extracted from the input text.
    """
    # Normalize text: lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    tokens = word_tokenize(text)

    # Removing the stopwords and lemmatize the text 
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stopwords.words("english")
    ]
    return clean_tokens


def build_model():
    """
    Build the ML pipeline and GridSearchCV

    Returns:
    model: GridSearchCV 
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=tokenize)),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'classifier__estimator__n_estimators': [50, 100],
        'classifier__estimator__min_samples_split': [2, 4]
    }

    # Use n_jobs=1 to avoid serialization issues in parallel processing
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3, n_jobs=1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the trained model on test data and print classification metrics.

    Args:
    model: Trained model
    X_test: Test data (features)
    Y_test: Test data (target labels)
    category_names: List of category names
    """
    Y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.

    Args:
    model: Trained model
    model_filepath: Filepath to save the pickle file
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
