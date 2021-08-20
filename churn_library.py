# library doc string
'''
A module to predict the company's churn

Author: Nícolas
Date: August 9, 2021
'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    # Create a numerical binary churn collumn
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Create a churn histogram
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('./images/eda/churn_hist.png')

    # Create a customer age histogram
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_hist.png')

    # Create a marital status bar chart
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_bar.png')

    # Create a total trans distribution plot
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig('./images/eda/total_trans_ct_dist.png')

    # Create a correlation heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/correlation_heatmap.png')


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features


    output:
            df: pandas dataframe with new columns for
    '''
    for category_name in category_lst:
        category_lst = []
        category_churn = df.groupby(category_name).mean()["Churn"]
        for val in df[category_name]:
            category_lst.append(category_churn.loc[val])
        df["{}_{}".format(category_name, "Churn")] = category_lst

    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df["Churn"]
    X = pd.DataFrame()
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"]
    X[keep_cols] = df[keep_cols]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    clf_report1 = classification_report(y_test,
                                        y_test_preds_rf,
                                        output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report1).iloc[:-1, :].T, annot=True)
    plt.title('random forest results\ntest results')
    plt.savefig('./images/results/rfc_report_test.png')
    plt.close()

    clf_report2 = classification_report(y_train,
                                        y_train_preds_rf,
                                        output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report2).iloc[:-1, :].T, annot=True)
    plt.title('random forest results\ntrain results')
    plt.savefig('./images/results/rfc_report_train.png')
    plt.close()

    clf_report3 = classification_report(y_test,
                                        y_test_preds_lr,
                                        output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report3).iloc[:-1, :].T, annot=True)
    plt.title('logistic regression results\ntest results')
    plt.savefig('./images/results/lrc_report_test.png')
    plt.close()

    clf_report4 = classification_report(y_train,
                                        y_train_preds_lr,
                                        output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report4).iloc[:-1, :].T, annot=True)
    plt.title('logistic regression results\ntrain results')
    plt.savefig('./images/results/lrc_report_train.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save plot in the output path
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(
        cv_rfc,
        X_test,
        "images/results/feature_importance.png")
    # save best model
    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")


if __name__ == '__main__':
    data = import_data('./data/bank_data.csv')
    perform_eda(data)
    encoded_df = encoder_helper(data,
                                ["Gender",
                                 "Education_Level",
                                 "Marital_Status",
                                 "Income_Category",
                                 "Card_Category"])
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        encoded_df)
    train_models(X_train, X_test, y_train, y_test)
