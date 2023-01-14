'''
Test module to test the churn_library

@author: Dan Rasmussen
@date: Jan 4, 2023
'''

import os
import sys
import traceback
import logging
import pytest
from churn_library import MLPipeline, Encoder

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Target coloumn
TARGET = 'Churn'

# Categorical columns of interest
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

# Quantitative columns of interest
QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

PARAM_GRID = {
    'n_estimators': [200],
    'max_features': ['auto'],
    'max_depth': [4],
    'criterion': ['entropy']
}


def clear_folder(path):
    '''
    clear files from a folder
    '''
    logging.info("Clear folder %s", path)
    for file in [file for file in os.listdir(path) if os.path.isfile(file)]:
        os.remove(os.path.join(path, file))


def test_setup():
    '''
    setup tests
    '''
    clear_folder("./images/eda")
    clear_folder("./images/results")
    clear_folder("./logs")
    clear_folder("./models")


def assert_help(test, message):
    '''
    assert helper so a line can be split between test and message
    '''
    assert test, message


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_df = import_data("./data/bank_data.csv")

        pytest.data_df = data_df
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture(scope="module", name="import_data")
def fixture_import_data():
    """
    Fixture - for import_data function defined in the MLPipeline
    """
    return MLPipeline.import_data


def test_categorical_to_binary(categorical_to_binary):
    '''
    test perform categorical_to_binary function
    '''
    try:
        encoded_df = categorical_to_binary(
            pytest.data_df, 'Attrition_Flag', 'Attrited Customer', TARGET)

        pytest.data_df = encoded_df
        logging.info("Testing categorical_to_binary: SUCCESS")
    except Exception as err:
        logging.error("Testing categorical_to_binary: ERROR")
        raise err

    try:
        assert TARGET in encoded_df, f"Target colum is missing: {TARGET}"
    except AssertionError as err:
        logging.error(
            "Testing categorical_to_binary: %s", str(err))
        raise err


@pytest.fixture(scope="module", name="categorical_to_binary")
def fixture_categorical_to_binary():
    """
    Fixture - for categorical_to_binary function defined in the Encoder
    """
    return Encoder.categorical_to_binary


def eda_file_exist(file):
    '''
    test helper for perform_eda
    '''
    return os.path.exists(f"./images/eda/{file}")


def test_perform_eda(perform_eda):
    '''
    test perform_eda
    '''
    try:
        perform_eda(pytest.data_df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: ERROR")
        logging.error(traceback.format_exc())
        raise err

    try:
        plot = "bar_plot_Marital_Status.png"
        assert eda_file_exist(plot), f"{plot} is missing"
        plot = "distribution_Total_Trans_Ct.png"
        assert eda_file_exist(plot), f"{plot} is missing"
        plot = "heatmap.png"
        assert eda_file_exist(plot), f"{plot} is missing"
        plot = "histogram_Customer_Age.png"
        assert eda_file_exist(plot), f"{plot} is missing"
        plot = "histogram_Churn.png"
        assert eda_file_exist(plot), f"{plot} is missing"

    except AssertionError as err:
        logging.error(
            "Testing perform_eda: %s", str(err))
        raise err


@pytest.fixture(scope="module", name="perform_eda")
def fixture_perform_eda():
    """
    Fixture - for import_data function defined in the MLPipeline
    """
    pipeline = MLPipeline(TARGET, CAT_COLUMNS, QUANT_COLUMNS, PARAM_GRID)
    return pipeline.perform_eda


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        encoded_df, encoded_columns = encoder_helper(
            pytest.data_df, CAT_COLUMNS, TARGET)

        pytest.data_df = encoded_df
        logging.info("Testing encoder_helper: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: ERROR")
        logging.error(traceback.format_exc())
        raise err

    try:
        assert_help(
            encoded_columns == [
                f"{column}_{TARGET}" for column in CAT_COLUMNS],
            f"Did not return expected encoded columns, {encoded_columns}"
        )
        for column in encoded_columns:
            assert column in encoded_df, f"Target colum is missing: {column}"
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: %s", str(err))
        raise err


@pytest.fixture(scope="module", name="encoder_helper")
def fixture_encoder_helper():
    """
    Fixture - for encoder_helper function defined in the Encoder
    """
    return Encoder.encoder_helper


def test_perform_feature_engineering(import_data, perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        data_df = import_data("./data/bank_data.csv")

        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data_df, TARGET)

        pytest.train = (x_train, y_train)
        pytest.test = (x_test, y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_feature_engineering: ERROR")
        logging.error(traceback.format_exc())
        raise err

    try:
        logging.info(x_train.shape)
        assert x_train.shape[0] > 0, "x_train row size is 0"
        assert x_train.shape[1] > 0, "x_train column size is 0"
        assert x_train.shape[0] == y_train.shape[
            0], f"x_train {x_train.shape[0]}, y_train {y_train.shape[0]} row size not equal"

        assert x_test.shape[0] > 0, "x_test row size is 0"
        assert x_test.shape[1] > 0, "x_test column size is 0"
        assert x_test.shape[0] == y_test.shape[
            0], f"x_test {x_test.shape[0]}, y_test {y_test.shape[0]} row size not equal"
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: %s", str(err))
        raise err


@pytest.fixture(scope="module", name="perform_feature_engineering")
def fixture_perform_feature_engineering():
    """
    Fixture - for perform_feature_engineering function defined in the MLPipeline
    """
    pipeline = MLPipeline(TARGET, CAT_COLUMNS, QUANT_COLUMNS, PARAM_GRID)
    return pipeline.perform_feature_engineering


def model_file_exist(file):
    '''
    test helper for test_train_models
    '''
    return os.path.exists(f"./models/{file}")


def results_file_exist(file):
    '''
    test helper for test_train_models
    '''
    return os.path.exists(f"./images/results/{file}")


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(pytest.train, pytest.test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: ERROR")
        logging.error(traceback.format_exc())
        raise err

    try:
        model = "logistic_model.pkl"
        assert model_file_exist(model), f"{model} is missing"
        model = "random forest_model.pkl"
        assert model_file_exist(model), f"{model} is missing"

        plot = "classification_report_logistic_regression.png"
        assert results_file_exist(plot), f"{plot} is missing"
        plot = "classification_report_random_forest.png"
        assert results_file_exist(plot), f"{plot} is missing"
        plot = "feature_importances.png"
        assert results_file_exist(plot), f"{plot} is missing"
        plot = "roc_curve.png"
        assert results_file_exist(plot), f"{plot} is missing"

    except AssertionError as err:
        logging.error(
            "Testing train_models: %s", str(err))
        raise err


@pytest.fixture(scope="module", name="train_models")
def fixture_train_models():
    """
    Fixture - for perform_feature_engineering function defined in the MLPipeline
    """
    pipeline = MLPipeline(TARGET, CAT_COLUMNS, QUANT_COLUMNS, PARAM_GRID)
    return pipeline.train_models


if __name__ == "__main__":
    # Run the tests
    sys.exit(pytest.main())
