# Disaster Response Pipeline Project
This project was completed as part of my Nanodegree program in Data Science. The primary objective is to build a Machine Learning model and a web application that processes and analyzes disaster-related messages, categorizing them into predefined categories such as requests for medical help, food or water supplies, infrastructure damage, and more.

The model is trained on real-world datasets provided by Figure Eight, containing thousands of labeled messages sent during various disasters. By leveraging Natural Language Processing (NLP) techniques and a robust classification pipeline, this project enables the automatic classification of text messages into multiple categories.

## Key Features
    ## 1. Machine Learning Pipeline:
Processes text data using NLP techniques like tokenization and lemmatization.
Builds a multi-output classifier to assign messages to one or more categories.
Employs hyperparameter tuning for optimal model performance.

    ## 2. Web Application:
Provides a user-friendly interface for organizations to input messages.
Displays categorized results and visualizations of message data.
Includes interactive visualizations to highlight the distribution of message genres and categories, as well as trends over time.
Impact and Practical Benefits
In disaster scenarios, rapid communication and response are critical to saving lives and reducing damage. Emergency responders and aid organizations often receive an overwhelming number of messages, making it difficult to manually sort and respond to them.

## This application provides the following key benefits:

Automation: It automates the classification of incoming messages, reducing the need for manual sorting.
Speed: Ensures that urgent needs, such as medical help or infrastructure repair, are identified and prioritized immediately.
Resource Allocation: Helps organizations allocate resources efficiently by highlighting the most common and urgent categories of requests.
Scalability: Capable of processing large volumes of messages quickly, making it suitable for widespread disaster scenarios.
By offering these functionalities, the project aims to improve disaster response efficiency and contribute to better outcomes for affected communities.

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
### Screenshots: 
    - Screenshot of the running model: 
![Screenshot 2024-12-11 111145](https://github.com/user-attachments/assets/d132dff1-602a-4e54-aab9-2f8a4c817d0b)

    - Screenshot of the Disaster Response Pipeline Project: 
![Screenshot 2024-12-11 152850](https://github.com/user-attachments/assets/19996e64-73c0-4758-922a-1ef51d316de1)

![Screenshot 2024-12-11 152806](https://github.com/user-attachments/assets/3a815135-0ebc-4181-8d68-b30ec2d706e7)

    - Screenshot of the Bar chart exists on the Project: 
![distributionMessageGenres](https://github.com/user-attachments/assets/1b6da375-79ff-4eb4-bf3d-7313060dc2dc)

    - Screenshot of Visual (1):  Distribution of Message Categories
![distributionMessageCategories](https://github.com/user-attachments/assets/651448a7-2853-40e0-a136-16d48fc72e09)

    - Screenshot of Visual (2):  Correlation Heatmap
![correlatoinHeatmap](https://github.com/user-attachments/assets/56cb2f6c-7726-4abb-8e92-884e57128271)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
