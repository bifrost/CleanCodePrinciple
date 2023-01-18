# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The purpose of this project is to split the churn_notebook.ipynb into production ready python code. The code have to be clean and efficient and has to meeting pep 8 guidelines. 

## Files and data description
The project consist of following files and directories:

### Files and data
**churn_notebook.ipynb**: The notebook that has to be converted inot production ready code.
**churn_library.py**: Library of functions to find customers who are likely to churn.
Classes  | Description
------------- | -------------
Encoder  | Utility class that implement encoder functions for a panda dataframe
ExploratoryDataAnalysis  | Class that implement exploratory data analysis functions (eda)
ClassificationReport  | Class to print and save a classification report for a model
TreeFeatureImportances  | Class that implement feature importances functions for tree models
RocCurvePlot  | Class that compare two models by plotting the roc curve plot
RandomForestGridSearchClassifier  | Class that implement GridSearch for RandomForest classifier
MLPipeline  | The MLPipeline class for the churn library
\_\_main\_\_  | Execute the MLPipeline for the setup in churn_notebook.ipynb
**churn_script_logging_and_tests.py**: Unit tests for the churn_library.py functions.
Function  | Description
------------- | -------------
\_\_main\_\_  | Run all tests using unittest framework and log the resuts.
**Guide.ipynb**: Guide for this exercise 
**pytest.ini**: Setup file for unittest framework
**Readme.md**: The readme file for this project
**requirements_py3.6.txt**: The Python requirements file for this project if running v3.6
**requirements_py3.8.txt**: The Python requirements file for this project if running v3.8

### Folders
**data**: Folder containing the input data bank_data.csv for the project 
**images**: Output folder for the eda analysis and prediction results
**logs**: Log folder for the churn_library.log
**models**: The model folder for the generted churn prediction models

## Running Files

### MLPipeline
To run the MLPipeline execute folowing in the console:
 `python churn_library.py`
 The following parameters can be changed in the file for the pipeline:
 - TARGET: The target coloumn
 - CAT_COLUMNS: Categorical columns of interest
 - QUANT_COLUMNS: Quantitative columns of interest

By running the pipeline the log will be shown in the console and following files will be created as result:

#### Eda
Folder images/eda:
**histogram_Churn.png**: Churn as univariate, quantitative plot
**histogram_Customer_Age.png**: Customer Age as univariate, quantitative plot
**bar_plot_Marital_Status.png**: Marital Status as univariate, categorical plot
**distribution_Total_Trans_Ct.png**: Total Trans Ct as distributions plot
**heatmap.png**: Pairwise correlation of columns as a bivariate plot

#### Prediction result
Folder images/results:
**classification_report_logistic_regression.png**: Classification report for logistic regression model
**classification_report_random_forest.png**: Classification report for best random forest model
**roc_curve.png**: The ROC curves of logistic regression - and random forest model
**feature_importances.png**: The feature importances of best random forest model
**shap_values.png**: Shape values of best random forest model

#### The models
Folder models:
**logistic_model.pkl**: The logistic regression model
**random forest_model.pkl**: The best random forest model

### Unit tests
To run the unit tests execute folowing in the console:
 `python churn_script_logging_and_tests.py`
or
`pytest`

By running the tests all files from MLPipeline will be created as well as a log file in the folder
**churn_library.log**: The results from the unit tests and the logs from the MLPipeline 

# Install
 To install the requirements for the project run:
**Workspace run**
`python -m pip install -r requirements_py3.6.txt`
**Local run**
`python -m pip install -r requirements_py3.8.txt`

# check
Run following to check the lint rules for the python files:
`pylint churn_library.py`
`pylint churn_script_logging_and_tests.py`
and to check how thwy conform to the PEP 8 style guide run (following will fix the files):
`autopep8 --in-place --aggressive --aggressive churn_library.py`
`autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py`