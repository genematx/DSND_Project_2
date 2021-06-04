# Udacity DS Nanodegree -- Project 2

 # ETL and ML Pipelines for Processing Disaster Response Messages

 The repository contains the code used in Project 2 of Udacity Data Science Nanodegree. The main goal of the project is to categorize text messages sent during a natural disaster into several classes to rapidly determine if any action is needed. The result is an on-line categorization system that on the title page displays brief statistics about the training dataset and allows the user to enter and classify a query message using the trained model.

 The motivation for this project is to gain practical skills in creating and deploying ETL and ML pipelines. The main highlights of the project include handling data in the form of CSV files and a relational database, feature engineering with natural language processing tools, and training a set of support vector machine classifiers.


 ## Installation <a name="installation"></a>

 The code is written and tested in Python 3.5. Additional libraries to be installed beyond the standard Anaconda distribution are:

- nltk 3.6.1 for natural language processing
- plotly 4.14.3 for displaying figures
- scikit-learn 0.20.0 for machine learning related tasks
- pandas 0.23.4 for handling the dataset
- sqlalchemy 1.3.24 for working with a relational database
- dill 0.3.0 for serializing and saving the model

The code has been developed and tested using the cited versions of the packages.


 ## File Descriptions <a name="files"></a>

  data/process_data.py - a script that loads the original messages and labels from .csv files (disaster_categories.csv and disaster_messages.csv), cleans the data, converts the categories into binary format and stores the resulting table in the SQLite database file, DisasterResponse.db.

  models/train_classifier.py - a script that loads the data from an SQLite database, creates an NLP-ML pipeline, and trains it with cross-validation. The output is a serialized model in the classifier.pkl file.

  app/run.py - the main backend script that handles the generation of the web pages. The templates for the pages are stored in the templates\ folder.

  ETL Pipeline Preparation.ipynb and ML Pipeline Preparation.ipynb - IPython notebooks used to build and test the pipelines (not required to run the main web app).


 ## How to run the app in Udacity Workspace

 The project is intended to be run in the Udacity Workspace. For this:

 1. Open a Terminal
 2. Enter the command `env | grep WORK` to find the current workspace variables
 3. In the app\ folder, run the command `python run.py`
 4. Open a new web browser window and go to the web address:
 `http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with relevant values from Step 1.


 ## Acknowledgements

 The disaster relief dataset for this project was kindly provided by <a href="https://en.wikipedia.org/wiki/Figure_Eight_Inc./">Figure Eight</a>, a machine learning and artificial intelligence company, currently a part of <a href="https://appen.com/">Appen</a>.

 The help from contributors on the <a href="https://www.udacity.com/">Udacity</a> discussion board is gratefully acknowledged.
