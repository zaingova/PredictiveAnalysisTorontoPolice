# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:38:21 2025

@author: phamdat225
"""

import pandas as pd

df = pd.read_csv("../data/hateCrimeData.csv")

print(df.head())
print(df.info())
df.describe(include="all")
#check mising values
df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(12,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Data Heatmap")
plt.show()

cols_to_drop = [
    'RACE_BIAS', 'ETHNICITY_BIAS', 'LANGUAGE_BIAS',
    'SEXUAL_ORIENTATION_BIAS', 'GENDER_BIAS'
]

df = df.drop(columns=cols_to_drop)

df['ARREST_MADE'].value_counts()

df['ARREST_MADE'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of ARREST_MADE")
plt.xlabel("Arrest Made?")
plt.ylabel("Count")
plt.show()

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

df = df.drop(columns=['OBJECTID'])
df = df.drop(columns=['REPORTED_YEAR'])

df['PRIMARY_OFFENCE'].value_counts().head()
df['LOCATION_TYPE'].value_counts().head()
df['NEIGHBOURHOOD_158'].value_counts().head()
df['RELIGION_BIAS'].value_counts().head()

#drop columns
df = df.drop(columns=['NEIGHBOURHOOD_140'], errors='ignore')
df = df.drop(columns=['REPORTED_YEAR'], errors='ignore')

#handle missing values
df['LOCATION_TYPE'] = df['LOCATION_TYPE'].fillna('Unknown')
df['ARREST_MADE'] = df['ARREST_MADE'].map({'Yes': 1, 'No': 0})
df['RELIGION_BIAS'] = df['RELIGION_BIAS'].fillna('Unknown')

#encoded values
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop('ARREST_MADE', axis=1)
y = df_encoded['ARREST_MADE']


#split to test and train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

corr = df_encoded.corr()['ARREST_MADE'].sort_values(ascending=False)
print(corr.head(20))