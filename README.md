# Disaster Response Pipeline Project
This project was completed as a requirement for my Nanodegree program in Data Science. The goal of this project is to build a Machine Learning model that processes and analyzes messages, classifying them into desired categories.
## Contents: 
1. app folder :
    - templates folder : contains the HTML files
    - run.py : main script to run the Flask Web App 
      
2. data folder :
     - disaster_categories.csv : csv file contains the categories database
     - disaster_messages.csv : csv file contains the messages database
     - DisasterResponse.db : the database that contains the DisasterResponse data
  
3. models folder :
    - train_classifier.py : contains the process of Machine Learning pipeline
    - classifier.pkl :  where I store the model after training and load it
  
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
