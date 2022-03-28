# ISOM3360 Group Project
This repository contains the source code for the business data mining course project (Group 23) (Spring 2022).  
Contributors (in alphabetical order):
> LAM, Ho Chit

> LEE, Ho Wan Owen

> LEE, Wai Chung

## Credit Card Defaultee Analysis
The goal of this project is to utilize data mining and machine learning methods to identify potential credit card defaultees.
The specific objectives are as follows:
- Help credit card issuers identify credit card defaultees
- Take extra precautions on people that are risky of delayed credit repayment

## Dataset description
The credit card defaultee dataset is sourced from Kaggle at
> https://www.kaggle.com/datasets/mishra5001/credit-card  

There are 3 files we obtain from this dataset. Due to course project restrictions, we use only the dataset in `application_data.csv` for analysis.  
This dataset consists of 120 feature columns (excluding ID), a complete column of data labels, and 308,000 data instances.

## Methodology
This project adopts classical data mining and machine learning models, specifically focusing on supervised learning as our objective is to predict potential defaultees.  
This project uses the programming language Python, due to its widespread popularity and collection of powerful open-source packages. Python libraries such as `numpy`, `pandas` and `scikit-learn` are utilized to perform data preprocessing, model training, prediction and evaluation.  
The entire project is split into 2 parts: data preprocessing and models analysis. Our team recommends that readers examine project code by following the order described in this documentation.

### Part 1: Data Preprocessing
One distinct characteristic of our dataset is that it is not divided into train or test sets. Therefore, extra steps are taken to manually divide and preprocess our dataset.  
Our data preprocessing workflow is as follows:

1. Explore features and characteristics of dataset
2. Drop columns of low data quality (e.g. large amounts of empty values)
3. Determine k columns to keep in the dataset (feature selection)
   - Performing elementary Lasso regression as a method of feature selection
4. Split into training and testing sets
5. Perform data cleaning
   - Dealing with missing values
6. Perform one-hot encoding on categorical values
7. Perform data standardization / normalization on continuous numerical values
8. Export preprocessed data to .csv files at [`./data_preprocessed/`](data_preprocessed)

### Part 2: Models Analysis
In this project, we utilize 4 supervised classification models. The order of model implementation and analysis is based on the sequence of teaching in ISOM3360 course syllabus.
#### Part 2.1: Decision Tree Classifier

#### Part 2.2: Logistic Regression

#### Part 2.3: Naive Bayes Classifier

#### Part 2.4: k Nearest Neighbours Classifier


## Repository structure
- [`README.md`](README.md): documentation for this course project
- [`data_raw/`](data_raw): directory that stores the raw `.csv` files obtained from the [Kaggle site](https://www.kaggle.com/datasets/mishra5001/credit-card)
  - `application_data.csv`: main dataset used
    - Not available in this repository due to large file size ([Kaggle link](https://www.kaggle.com/datasets/mishra5001/credit-card?select=application_data.csv))
  - [`columns_description.csv`](data_raw/columns_description.csv): explains the columns/features in `application_data.csv` and `previous_application.csv`
    - ([Kaggle link](https://www.kaggle.com/datasets/mishra5001/credit-card?select=columns_description.csv))
  - `previous_application.csv`: dataset with less features, not used in this project due to project constraints
    - Not available in this repository due to large file size ([Kaggle link](https://www.kaggle.com/datasets/mishra5001/credit-card?select=previous_application.csv))
- [`preprocessing.ipynb`](preprocess.ipynb): performs all universal data preprocessing tasks
- [`data_preprocessed/`](data_preprocessed): directory that stores the preprocessed data in `.csv` files
  - `train.csv`: stores preprocessed training data
  - `test.csv`: stores preprocessed test data
- [`models/`](models): directory that stores `.ipynb` notebooks containing different models
  - [`decision_tree.ipynb`](models/decision_tree.ipynb): Decision Tree Classifier
  - [`logistic_regression.ipynb`](models/logistic_regression.ipynb): Logistic Regression
  - [`naive_bayes.ipynb`](models/naive_bayes.ipynb): Naive Bayes Classifier
  - [`k_nearest_neighbours.ipynb`](models/k_nearest_neighbours.ipynb): k Nearest Neighbours Classifier
- [`utils/`](utils): directory that stores reusable codes into self-defined Python modules stored in `.py` files
  - [`analysis.py`](utils/analysis.py): contains helper functions to analyze predicted results and model performance


## This is the end of documentation for ISOM3360 Group Project.
