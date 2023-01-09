# library doc string
'''
Module to ...

@author: Dan Rasmussen
@date: Jan 4, 2023
'''

# import libraries
import os
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
import logging
from re import sub

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


DATA_PATH = r"./data"
IMAGE_PATH = r"./images"
MODEL_PATH = r"./models"


class Encoder:

    def __init__(self, cat_columns, quant_columns):
        self.cat_columns = cat_columns
        self.quant_columns = quant_columns

    def categorical_to_binary(self, input_df, column, true_category, response):
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
            f'add binary column {response}, derieved from: {column}, true_category: "{true_category}"')

        input_df[response] = input_df[column].apply(
            lambda val: 1 if val == true_category else 0)

        return input_df

    def encoder_helper(
            self,
            input_df,
            category_list,
            response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category (mean of the response) - associated with cell 15 from the notebook

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

            logging.info(f'add column {column}')
            input_df[column] = [groups[val]
                                for val in input_df[category]]

            encoded_columns.append(column)

        return input_df, encoded_columns


class ExploratoryDataAnalysis:

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

    def __init_plot(self):
        plt.figure(figsize=(20, 10))

    def __save_plot(self, file_name):
        file_path = f'{self.data_path}/{file_name}'
        logging.info(f'save eda plot {file_path}')
        plt.savefig(file_path)
        plt.close()


        
class RandomForestGridSearchClassifier:
    '''
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    '''

    param_grid = {
        'n_estimators': [200],
        'max_features': ['auto'],
        'max_depth': [4],
        'criterion': ['entropy']
    }

    def fit(self, x_train, y_train):
        # grid search
        rfc = RandomForestClassifier(random_state=42)

        cv_rfc = GridSearchCV(
            estimator=rfc,
            param_grid=RandomForestGridSearchClassifier.param_grid,
            cv=5)
        cv_rfc.fit(x_train, y_train)

        return cv_rfc.best_estimator_


class ClassificationReport:

    def __init__(self, data_path):
        self.data_path = data_path

    def save_report(
            self,
            model_name,
            y_train,
            y_test,
            y_train_preds,
            y_test_preds):

        self.__plot_report(
            model_name,
            y_train,
            y_test,
            y_train_preds,
            y_test_preds)

        file_name = self.__snake_case(f'classification report {model_name}')
        self.__save(file_name)

    def __plot_report(
            self,
            model_name,
            y_train,
            y_test,
            y_train_preds,
            y_test_preds):

        plt.rc('figure', figsize=(5, 7))

        plt.text(0.01, 1.1, str(f'{model_name} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!

        plt.text(0.01, 0.6, str(f'{model_name} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')

    def __save(self, file_name):
        file_path = f'{self.data_path}/{file_name}.png'
        logging.info(f'save classification report {file_path}')
        plt.savefig(file_path)
        plt.close()

    def __snake_case(self, name):
        return '_'.join(sub('([A-Z][a-z]+)',
                            r' \1',
                            sub('([A-Z]+)',
                                r' \1',
                                name.replace('-',
                                             ' '))).split()).lower()


class FeatureImportances:

    def __init__(self, data_path):
        self.data_path = data_path

    def save_shap_values(self,
            model,
            x_data):
        '''
        save shap values for a tree model
        input:
                input_df (pd.DataFrame): pandas dataframe
                column (str): name of the column

        output:
                None
        '''
        self.__init_plot()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_data)
        shap.summary_plot(shap_values, x_data, plot_type="bar",show=False)

        self.__save_plot(f'shap_values.png')
        
    def save_feature_importances(self,
            model,
            x):
        '''
        save feature importances for a tree model sorted by importances
        input:
                x (pd.DataFrame): pandas dataframe of the model's x values
                column (str): name of the column

        output:
                None
        '''
        self.__init_plot()

        # Calculate feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [x.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(x.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(x.shape[1]), names, rotation=90);

        self.__save_plot(f'feature_importances.png')

    def __init_plot(self):
        plt.figure(figsize=(15, 8))

    def __save_plot(self, file_name):
        file_path = f'{self.data_path}/{file_name}'
        logging.info(f'save feature importances plot {file_path}')
        plt.savefig(file_path)
        plt.close()
        
        
class MLPipeline():

    target = 'Churn'

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
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

    def __init__(self):

        self.encoder = Encoder(
            MLPipeline.cat_columns,
            MLPipeline.quant_columns)
        self.eda = ExploratoryDataAnalysis(f'{IMAGE_PATH}/eda')
        self.report = ClassificationReport(f'{IMAGE_PATH}/results')

    def import_data(self, input_path):
        '''
        returns dataframe for the csv found at input_path

        input:
                input_path (str): a path to the csv
        output:
                (pd.DataFrame): pandas dataframe
        '''
        logging.info(f'import data from {input_path}')
        return pd.read_csv(input_path)

    def perform_feature_engineering(self, input_df, response):
        '''
        ???
        
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
        encoded_df = self.encoder.categorical_to_binary(
            input_df, 'Attrition_Flag', 'Attrited Customer', response)

        encoded_df, encoded_columns = self.encoder.encoder_helper(
            encoded_df, MLPipeline.cat_columns, response)

        self.encoded_columns = encoded_columns

        keep_columns = MLPipeline.quant_columns + encoded_columns
        logging.info(f'x: {keep_columns}')
        logging.info(f'y: {response}')

        x = encoded_df[keep_columns]
        y = encoded_df[response]

        # NOTE: I would have separated feature engineering from
        # train_test_split

        # This cell may take up to 15-20 minutes to run
        # train test split
        logging.info(f'splitting data...')
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42)

        logging.info(f'x_train: {x_train.shape}')
        logging.info(f'x_test: {x_test.shape}')
        logging.info(f'y_train: {y_train.shape}')
        logging.info(f'y_test: {y_test.shape}')

        return x_train, x_test, y_train, y_test

    def perform_eda(self, input_df):
        '''
        perform Exploratory Data Analysis (eda) on df and save figures to images folder

        input:
                input_df (pd.DataFrame): pandas dataframe

        output:
                None
        '''
        logging.info(f'perform eda')

        # TODO: save null values and stats

        self.eda.save_histogram(input_df, 'Churn')
        self.eda.save_histogram(input_df, 'Customer_Age')
        self.eda.save_bar_plot(input_df, 'Marital_Status')
        self.eda.save_distribution_plot(input_df, 'Total_Trans_Ct')

        # keep column order, list(set(x) - set(y)) does not keep order
        eda_colums = [
            c for c in input_df.columns if c not in self.encoded_columns]
        self.eda.save_heatmap(input_df[eda_colums])

    def train_models(
            self,
            x_train,
            x_test,
            y_train,
            y_test):
        '''
        train, store model results: images + scores, and store models
        
        input:
                  x_train (pd.DataFrame): X training data
                  x_test (pd.DataFrame): X testing data
                  y_train (pd.Series): y training data
                  y_test (pd.Series): y testing data
        output:
                  None
        '''
        # grid search
        rfc = RandomForestGridSearchClassifier()
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        classifier_list = [('rfc', rfc), ('logistic', lrc)]

        y_train_preds = {}
        y_test_preds = {}

        # train and predict each classifier
        for item in classifier_list:
            name, classifier = item

            logging.info(f'train classifier {name}')
            model = classifier.fit(x_train, y_train)

            model_path = f'{MODEL_PATH}/{name}_model.pkl'
            logging.info(f'save model {model_path}')
            joblib.dump(model, model_path)

            logging.info(f'predict y_train {name}')
            y_train_preds[name] = model.predict(x_train)

            logging.info(f'predict y_test {name}')
            y_test_preds[name] = model.predict(x_test)

        # Too much responsibility for this function
        # the plot should be moved to execute
        
        self.classification_report_image(
            y_train,
            y_test,
            y_train_preds['logistic'],
            y_train_preds['rfc'],
            y_test_preds['logistic'],
            y_test_preds['rfc'])
        
        # feature_importance_plot
        # roc_curve

    
    
    def classification_report_image(self, y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train (pd.Series): training response values
                y_test (pd.Series):  test response values
                y_train_preds_lr (pd.Series): training predictions from logistic regression
                y_train_preds_rf (pd.Series): training predictions from random forest
                y_test_preds_lr (pd.Series): test predictions from logistic regression
                y_test_preds_rf (pd.Series): test predictions from random forest

        output:
                 None
        '''
        self.report.save_report(
            'Logistic Regression',
            y_train,
            y_test,
            y_train_preds_lr,
            y_test_preds_lr)
        
        self.report.save_report(
            'Random Forest',
            y_train,
            y_test,
            y_train_preds_rf,
            y_test_preds_rf)

    def feature_importance_plot(
            self,
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
        pass

    def execute(self, data_path):
        bank_data_df = self.import_data(data_path)[0:500]

        x_train, x_test, y_train, y_test = self.perform_feature_engineering(
            bank_data_df, MLPipeline.target)

        self.perform_eda(bank_data_df)

        self.train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    pipeline = MLPipeline()
    pipeline.execute(r"./data/bank_data.csv")
