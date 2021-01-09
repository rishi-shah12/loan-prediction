import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Training Dataset
train = pd.read_csv('Dataset/train_ctrUa4K.csv')
# Making a copy of the original data
train_original = train.copy()

# Testing dataset
test = pd.read_csv('Dataset/test_lAUu6dG.csv')
# Training Dataset
test_original = test.copy()

# Filling in the missing data with the mode from their respective columns (train dataset)
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

# Filling in the empty loan amounts with the median loan amount in the column (train dataset)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# Filling in the missing data with the mode from their respective columns (test dataset, still using the train mode values)
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Married'].fillna(train['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

# Filling in the empty loan amounts with the median loan amount in the column (test dataset, still using the test median value)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# Dropping Loan_ID as that doesn't affect the
train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

train['Total_Income'] = train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income'] = test['ApplicantIncome']+test['CoapplicantIncome']

train['EMI'] = train['LoanAmount']/train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount']/test['Loan_Amount_Term']

train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

# Putting the individual results in its own dataset
X = train.drop('Loan_Status', 1)
y = train.Loan_Status

# Convert the variables with words to numbers for use in the algorithm
X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Getting the train test split data to run on the machine learning model
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)

# Creating and training the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression()

# Predicting the dataset figures
pred_cv = model.predict(x_cv)
print(accuracy_score(y_cv,pred_cv)*100)


