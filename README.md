# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
- This project implements a model to identify credit card customers that are most likely to churn. The project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package also have the flexibility of being run interactively or from the command-line interface (CLI)

## Running Files
- The churn_library.py script runs a dataset with some feateures from a company. The script does all the steps from the feature engineering to training the best model and saving it. The script run the dataset feature enginering and saves some features graphs. The algorithms used for the model training were a Random Forest Classifier using GridSearchCV from Scikit-Learning and another used a Logistic Regression. The models predict the customer churn and output confusion matrix graphs with the results. It also saves these two models in pickle format. To run this script use the following command line:

  - python churn_library.py

- It was also created the churn_script_logging_and_tests.py that tests all the fuctions of the churn_library.py script. When it is required to run it you should use the following command line:

  - pytest churn_script_logging_and_tests.py

## Files in the Repository
The following represents the schema of directories with each file inside it:

- data
   - bank_data.csv
- images
  - eda
    - churn_hist.png
    - correlation_heatmap.png
    - customer_age_hist.png
    - marital_status_bar.png
    - total_trans_ct_dist.png
  - results
    - rfc_report_test.png
    - rfc_report_train.png
    - lrc_report_test.png
    - lrc_report_train.png
    - feature_importance.png
- logs
  - churn_library.log
- models
  - logistic_model.pkl
  - rfc_model.pkl
- churn_library.py
- churn_notebook.ipynb
- churn_script_logging_and_tests.py
- README.md

In the data folder is located the dataset provided to predict the company's churn.\
In the EDA folder is made the exploratory data analysis with histograms, bar charts, heatmaps and distributions.\
In the results forder there is the the metrics reports for the random forest classifier and logistic classifier and the feature importances.\
In the logs folder there is the churn_library.log that is created when run the unit test script churn_script_logging_and_tests.py.\
In the models folder is saved the best models for the random forest classifier and logistic classifier algorithms in pickle format.\
The churn_library.py is the script that does all the steps from the feature engineering to training the best model and saving it.\
The churn_script_logging_and_tests.py is the one used for the unit test to be run using pytest.

## Required modules 
The following models are required to run the scripts:

- shap
- joblib
- pandas 
- numpy 
- matplotlib 
- seaborn
- sklearn
- logging
- pytest
- joblib






