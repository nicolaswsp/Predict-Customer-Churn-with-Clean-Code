'''
Testing & Logging for churn_library.py script

Author: Nícolas
Date: August 9, 2021
'''
import os
import logging
import pytest
import joblib
import churn_library

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name='df_initial')
def df_initial():
    """
    initial dataframe fixture
    """
    try:
        df_initial = churn_library.import_data("./data/bank_data.csv")
        logging.info("Initial dataframe fixture creation: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Initial fixture creation: The file wasn't found")
        raise err
    return df_initial


@pytest.fixture(name='df_encoded')
def df_encoded(df_initial):
    """
    encoded dataframe fixture
    """
    try:
        df_encoded = churn_library.encoder_helper(
            df_initial, ["Gender",
                         "Education_Level",
                         "Marital_Status",
                         "Income_Category",
                         "Card_Category"])
        logging.info("Encoded dataframe fixture creation: SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoded dataframe fixture creation: It was not possible to encode some columns")
        raise err
    return df_encoded


@pytest.fixture(name='feature_engineering')
def feature_engineering(df_encoded):
    """
    feature_engineering fixtures
    """
    try:
        X_train, X_test, y_train, y_test = churn_library.perform_feature_engineering(
            df_encoded)
        logging.info("Feature engineering fixture creation: SUCCESS")
    except BaseException as err:
        logging.error(
            "Feature engineering fixture creation: Features lengths mismatch")
        raise err
    return X_train, X_test, y_train, y_test


def test_import(df_initial):
    '''
    test data import
    '''
    try:
        assert df_initial.shape[0] > 0
        assert df_initial.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_initial):
    '''
    test perform eda function
    '''
    churn_library.perform_eda(df_initial)
    images_list = [
        "churn_hist",
        "customer_age_hist",
        "marital_status_bar",
        "total_trans_ct_dist",
        "correlation_heatmap"]
    for image_name in images_list:
        try:
            with open("./images/eda/{}.png".format(image_name), 'r'):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing perform_eda: Some of the images is missing")
            raise err


def test_encoder_helper(df_encoded):
    '''
    test encoder helper
    '''
    try:
        assert df_encoded.shape[0] > 0
        assert df_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have rows and columns")
        raise err

    try:
        encoded_columns = ["Gender",
                           "Education_Level",
                           "Marital_Status",
                           "Income_Category",
                           "Card_Category"]
        for column in encoded_columns:
            assert column in df_encoded
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")
    return df_encoded


def test_perform_feature_engineering(feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train = feature_engineering[0]
        X_test = feature_engineering[1]
        y_train = feature_engineering[2]
        y_test = feature_engineering[3]
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Features lengths mismatch")
        raise err
    return feature_engineering


def test_train_models(feature_engineering):
    '''
    test train_models
    '''
    churn_library.train_models(
        feature_engineering[0],
        feature_engineering[1],
        feature_engineering[2],
        feature_engineering[3])
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The file was not found")
        raise err
    images_list = ["rfc_report_test",
                   "rfc_report_train",
                   "lrc_report_test",
                   "lrc_report_train",
                   "feature_importance"]
    for image_name in images_list:
        try:
            with open("./images/results/{}.png".format(image_name), 'r'):
                logging.info("Testing train_models: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing train_models:  Some of the images is missing")
            raise err


if __name__ == "__main__":
    pass
