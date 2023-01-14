# library doc string
'''
Module that implement several utility classes for customer churn prediction.
The main class MLPipeline is used to execute the ML Pipeline.

@author: Dan Rasmussen
@date: Jan 4, 2023
'''

# import libraries
import os
import logging
from re import sub
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


DATA_PATH = r"./data"
IMAGE_PATH = r"./images"
MODEL_PATH = r"./models"


class Encoder:
    '''
    Utility class that implement encoder functions for a panda dataframe

    Note: Should have its own file
    '''

    @staticmethod
    def categorical_to_binary(input_df, column, true_category, response):
        '''
        Convert a categorical column into a new binary column

        input:
                input_df (pd.DataFrame): pandas dataframe
                column (str): column name
                true_category (str): the category that converts to 1
                response (str): string of response name

        output:
                (pd.DataFrame): pandas dataframe
        '''
        logging.info(
            f'add binary column {response}, ' +
            f'derieved from: {column}, true_category: "{true_category}"')

        input_df[response] = input_df[column].apply(
            lambda val: 1 if val == true_category else 0)

        return input_df

    @staticmethod
    def encoder_helper(
            input_df,
            category_list,
            response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category (mean of the response) - associated
        with cell 15 from the notebook

        Note: Normally I would have renamed this method to categoricalTargetencoding but
        because it is a predefined method for this exercise I do not change the name.

        input:
                input_df (pd.DataFrame): pandas dataframe
                category_list (list[str]): list of columns that contain categorical features
                response (str): string of response name [optional argument that could
                be used for naming variables or index y column]

        output:
                pd.DataFrame: pandas dataframe with new columns for
        '''
        encoded_columns = []
        for category in category_list:
            # do only calculate mean of Churn and skip other columns
            groups = input_df[[category, response]].groupby(category).mean()[
                response]

            column = f'{category}_{response}'

            logging.info('add column %s', column)
            input_df[column] = [groups[val]
                                for val in input_df[category]]

            encoded_columns.append(column)

        return input_df, encoded_columns


class ExploratoryDataAnalysis:
    '''
    Utility class that implement exploratory data analysis functions (eda)

    Note: Should have its own file
    '''

    def __init__(self, data_path):
        self.data_path = data_path

    def save_histogram(self, input_df, column):
        '''
        save histogram for the column in input_df
        input:
                input_df (pd.DataFrame): pandas dataframe
                column (str): name of the column

        output:
                None
        '''
        self.__init_plot()

        input_df[column].hist()

        self.__save_plot(f'histogram_{column}.png')

    def save_bar_plot(self, input_df, column):
        '''
        save bar plot for the column in input_df
        input:
                input_df (pd.DataFrame): pandas dataframe
                column (str): name of the column

        output:
                None
        '''
        self.__init_plot()

        input_df[column].value_counts('normalize').plot(kind='bar')

        self.__save_plot(f'bar_plot_{column}.png')

    def save_distribution_plot(self, input_df, column):
        '''
        save distribution plot for the column in input_df
        input:
                input_df (pd.DataFrame): pandas dataframe
                column (str): name of the column

        output:
                None
        '''
        self.__init_plot()

        # distplot is deprecated. Use histplot instead
        # sns.distplot(df['Total_Trans_Ct']);
        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
        # using a kernel density estimate
        sns.histplot(input_df[column], stat='density', kde=True)

        self.__save_plot(f'distribution_{column}.png')

    def save_heatmap(self, input_df):
        '''
        Compute pairwise correlation of columns
        in input_df and save it as a heatmap
        input:
                input_df (pd.DataFrame): pandas dataframe

        output:
                None
        '''
        self.__init_plot()

        sns.heatmap(input_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)

        self.__save_plot(r'heatmap.png')

    @staticmethod
    def __init_plot():
        plt.figure(figsize=(20, 10))

    def __save_plot(self, file_name):
        file_path = f'{self.data_path}/{file_name}'
        logging.info('save eda plot %s', file_path)
        plt.savefig(file_path)
        plt.close()


class ClassificationReport:
    '''
    Utility class to print and save a classification report for a model

    Note: Should have its own file
    '''

    def __init__(self, data_path):
        self.data_path = data_path

    def save_report(
            self,
            model_name,
            traint,
            test):
        '''
        save classification report

        input:
                model_name (str): the model name
                y_train (pd.Series): y training data (ground truth)
                y_test (pd.Series): y test data (ground truth)
                y_train_preds (pd.Series): y prediction of training data
                y_test_preds (pd.Series): y prediction of test data

        output:
                None
        '''

        self.__plot_report(
            model_name,
            traint,
            test)

        file_name = self.snake_case(f'classification report {model_name}')
        self.__save(file_name)

    @staticmethod
    def __plot_report(
            model_name,
            train,
            test):
        '''
        helper function to plot the classification report

        input:
                model_name (str): the model name
                y_train (pd.Series): y training data (ground truth)
                y_test (pd.Series): y test data (ground truth)
                y_train_preds (pd.Series): y prediction of training data
                y_test_preds (pd.Series): y prediction of test data

        output:
                None
        '''
        y_train, y_train_preds = train
        y_test, y_test_preds = test

        plt.rc('figure', figsize=(10, 5))

        plt.text(0.01, 0.5, str(f'{model_name} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_train, y_train_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 1.05, str(f'{model_name} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.60, str(
                classification_report(
                    y_test, y_test_preds)), {
                'fontsize': 10}, fontproperties='monospace')

        plt.axis('off')

    def __save(self, file_name):
        file_path = f'{self.data_path}/{file_name}.png'
        logging.info('save classification report %s', file_path)
        plt.savefig(file_path)
        plt.close()

    @staticmethod
    def snake_case(name):
        '''
        helper function to convert a string to snake case

        input:
                name (str): the string to convert

        output:
                (str): converted string
        '''
        return '_'.join(sub('([A-Z][a-z]+)',
                            r' \1',
                            sub('([A-Z]+)',
                                r' \1',
                                name.replace('-',
                                             ' '))).split()).lower()


class TreeFeatureImportances:
    '''
    Utility class that implement feature importances functions for tree models

    Note: Should have its own file
    '''

    def __init__(self, data_path):
        self.data_path = data_path

    def save_shap_values(self,
                         model,
                         x_data):
        '''
        save shap values for a tree model
        input:
                model (DecisionTreeClassifier): a tree model
                x_data (pd.DataFrame): x test data

        output:
                None
        '''
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_data)
        shap.summary_plot(
            shap_values,
            x_data,
            plot_type="bar",
            show=False,
            plot_size=(
                20,
                20))

        self.__save_plot('shap_values.png')

    def save_feature_importances(self,
                                 model,
                                 x_data):
        '''
        save feature importances for a tree model sorted by importances
        input:
                model (DecisionTreeClassifier): a tree model
                x_data (pd.DataFrame): x data

        output:
                None
        '''
        plt.figure(figsize=(20, 20))

        # Calculate feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [x_data.columns[i] for i in indices]

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(x_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(x_data.shape[1]), names, rotation=90)

        self.__save_plot('feature_importances.png')

    def __save_plot(self, file_name):
        file_path = f'{self.data_path}/{file_name}'
        logging.info('save feature importances plot %s', file_path)
        plt.savefig(file_path)
        plt.close()


class RocCurvePlot:
    '''
    Utility class that compare two models by plotting the roc curve plot

    Note: Should have its own file
    '''

    def __init__(self, data_path):
        self.__data_path = data_path

    def get_data_path(self):
        '''
        get data path

        return
            (str) data path
        '''
        return self.__data_path

    def save_roc_curves(self, model_1, model_2, x_test, y_test):
        '''
        save two models roc curve plot in one plot
        input:
                model_1 (DecisionTreeClassifier): a tree model
                model_2 (DecisionTreeClassifier): a tree model
                x_test (pd.DataFrame): x test data
                y_test (pd.Series): y test data

        output:
                None
        '''
        plot = plot_roc_curve(model_1, x_test, y_test)

        plt.figure(figsize=(15, 8))
        axes = plt.gca()
        plot_roc_curve(model_2, x_test, y_test, ax=axes, alpha=0.8)
        plot.plot(ax=axes, alpha=0.8)

        self.__save_plot('roc_curve.png')

    def __save_plot(self, file_name):
        file_path = f'{self.__data_path}/{file_name}'
        logging.info('save roc curve plot %s', file_path)
        plt.savefig(file_path)
        plt.close()


class RandomForestGridSearchClassifier:
    '''
    Utility class that implement GridSearch for RandomForest classifier

    Implement a sklearn fit method so it can be used as a classifier
    '''

    def __init__(self, param_grid):
        '''
        constructor

        input:
            param_grid: grid search parameters
        '''
        self._param_grid = param_grid

    def get_param_grid(self):
        '''
        getter for param_grid

        output:
            param_grid: grid search parameters
        '''
        return self._param_grid

    def fit(self, x_train, y_train):
        '''
        fit data to RandomForestClassifier by using GridSearch and
        returns the best estimator

        input:
                x_train (pd.DataFrame): x training data
                y_train (pd.Series): y training data

        output:
                (RandomForestClassifier): the best estimator
        '''

        # grid search
        rfc = RandomForestClassifier(random_state=42)

        cv_rfc = GridSearchCV(
            estimator=rfc,
            param_grid=self._param_grid,
            cv=5)
        cv_rfc.fit(x_train, y_train)

        return cv_rfc.best_estimator_


class MLPipeline():
    '''
    The MLPipeline class for the churn library
    '''

    def __init__(self, target, cat_columns, quant_columns, param_grid):
        '''
        initialization of ML Pipeline

        input:
                target (str): Target column name
                cat_columns (list[str]): Categorical columns of interest
                quant_columns (list[str]): Quantitative columns of interest
                param_grid (Dictionary): Parameters for tree grid search
        '''
        self.target = target
        self.cat_columns = cat_columns
        self.quant_columns = quant_columns
        self.param_grid = param_grid

        self.encoded_columns = []

    @staticmethod
    def import_data(input_path):
        '''
        returns dataframe for the csv found at input_path

        input:
                input_path (str): a path to the csv
        output:
                (pd.DataFrame): pandas dataframe
        '''
        logging.info('import data from %s', input_path)
        return pd.read_csv(input_path)

    def perform_feature_engineering(self, input_df, response):
        '''
        perform feature engineering on input_df
        1. binarize Attrition_Flag column and add it as response to input_df,
           value 'Attrited Customer' is 1 else 0
        2. target encoding categorical columns 'cat_columns' and add it to input_df
           prefixed with response

        input:
                  input_df (pd.DataFrame): pandas dataframe
                  response (str): string of response name [optional argument that could
                  be used for naming variables or index y column]

        output:
                  x_train (pd.DataFrame): x training data
                  x_test (pd.DataFrame): x testing data
                  y_train (pd.Series): y training data
                  y_test (pd.Series): y testing data
        '''
        encoder = Encoder()

        encoded_df = encoder.categorical_to_binary(
            input_df, 'Attrition_Flag', 'Attrited Customer', response)

        encoded_df, encoded_columns = encoder.encoder_helper(
            encoded_df, self.cat_columns, response)

        self.encoded_columns = encoded_columns

        keep_columns = self.quant_columns + encoded_columns
        logging.info('x: %s', keep_columns)
        logging.info('y: %s', response)

        encoded_x = encoded_df[keep_columns]
        encoded_y = encoded_df[response]

        # NOTE: I would have separated feature engineering from
        # train_test_split

        # This cell may take up to 15-20 minutes to run
        # train test split
        logging.info('splitting data...')
        x_train, x_test, y_train, y_test = train_test_split(
            encoded_x, encoded_y, test_size=0.3, random_state=42)

        logging.info('x_train: %s', x_train.shape)
        logging.info('x_test: %s', x_test.shape)
        logging.info('y_train: %s', y_train.shape)
        logging.info('y_test: %s', y_test.shape)

        return x_train, x_test, y_train, y_test

    def perform_eda(self, input_df):
        '''
        perform Exploratory Data Analysis (eda) on input_df and save figures to images folder

        input:
                input_df (pd.DataFrame): pandas dataframe

        output:
                None
        '''
        logging.info('perform eda')
        eda = ExploratoryDataAnalysis(f'{IMAGE_PATH}/eda')

        eda.save_histogram(input_df, 'Churn')
        eda.save_histogram(input_df, 'Customer_Age')
        eda.save_bar_plot(input_df, 'Marital_Status')
        eda.save_distribution_plot(input_df, 'Total_Trans_Ct')

        # keep column order, list(set(x) - set(y)) does not keep order
        eda_colums = [
            c for c in input_df.columns if c not in self.encoded_columns]
        eda.save_heatmap(input_df[eda_colums])

    def train_models(
            self,
            train,
            test):
        '''
        train, store model results: images + scores, and store models

        input:
                  train (tuple(pd.DataFrame, pd.Series)): x, y training data
                  test (tuple(pd.DataFrame, pd.Series)): x, y test data
        output:
                  None
        '''
        # Note: I would have prefered to return the model and maybe the predicted values
        # to reduce responsibility of this function

        # grid search
        rfc = RandomForestGridSearchClassifier(self.param_grid)

        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        classifier_list = [('logistic', lrc), ('random forest', rfc)]

        models = {}
        y_train_preds = {}
        y_test_preds = {}

        # train and predict each classifier
        for item in classifier_list:
            name, classifier = item

            logging.info('train classifier %s', name)
            models[name] = classifier.fit(train[0], train[1])

            model_path = f'{MODEL_PATH}/{name}_model.pkl'
            logging.info('save model %s', model_path)
            joblib.dump(models[name], model_path)

            logging.info('predict y_train %s', name)
            y_train_preds[name] = models[name].predict(train[0])

            logging.info('predict y_test %s', name)
            y_test_preds[name] = models[name].predict(test[0])

        # Too much responsibility for this function
        # the plot should be moved to execute

        self.plot_results(models, (train, y_train_preds), (test, y_test_preds))

    def plot_results(
            self,
            models,
            train,
            test):
        '''
        plot classification_report, feature_importance and roc_curves
        '''
        (_, y_train), y_train_preds = train
        (x_test, y_test), y_test_preds = test

        self.classification_report_image(
            (y_train, y_test),
            (y_train_preds['logistic'], y_test_preds['logistic']),
            (y_train_preds['random forest'], y_test_preds['random forest']))

        # feature importance plot
        self.feature_importance_plot(
            models['random forest'],
            x_test,
            f'{IMAGE_PATH}/results')

        # roc_curve
        roc = RocCurvePlot(f'{IMAGE_PATH}/results')
        roc.save_roc_curves(
            models['logistic'],
            models['random forest'],
            x_test,
            y_test)

    @staticmethod
    def classification_report_image(target, logistic, random_forest):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                target (tuple(pd.Series, pd.Series)): target values from train & test data
                logistic (tuple(pd.Series, pd.Series)): predicted values from Logistic Regression
                random_forest (tuple(pd.Series, pd.Series)): predicted values from Random Forest

        output:
                 None
        '''
        report = ClassificationReport(f'{IMAGE_PATH}/results')

        report.save_report(
            'Logistic Regression',
            (target[0], logistic[0]),
            (target[1], logistic[1]))

        report.save_report(
            'Random Forest',
            (target[0], random_forest[0]),
            (target[1], random_forest[1]))

    @staticmethod
    def feature_importance_plot(
            model,
            x_data,
            output_path):
        '''
        creates and stores the feature importances in output_path

        input:
                model (BaseEstimator): model object containing feature_importances_
                x_data (pd.DataFrame): pandas dataframe of X values
                output_path (str): path to store the figure

        output:
                 None
        '''
        importances = TreeFeatureImportances(output_path)
        importances.save_shap_values(model, x_data)
        importances.save_feature_importances(model, x_data)

    def execute(self, data_path):
        '''
        execute the ML pipeline

        input:
                data_path (str): path to the input data file

        output:
                 None
        '''
        bank_data_df = self.import_data(data_path)

        x_train, x_test, y_train, y_test = self.perform_feature_engineering(
            bank_data_df, self.target)

        self.perform_eda(bank_data_df)

        self.train_models((x_train, y_train), (x_test, y_test))


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

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
    # PARAM_GRID = {
    #    'n_estimators': [200, 500],
    #    'max_features': ['auto', 'sqrt'],
    #    'max_depth' : [4,5,100],
    #    'criterion' :['gini', 'entropy']
    # }

    pipeline = MLPipeline(TARGET, CAT_COLUMNS, QUANT_COLUMNS, PARAM_GRID)
    pipeline.execute(r"./data/bank_data.csv")
