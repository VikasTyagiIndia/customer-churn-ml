# Importing libraries
import pandas as pd
import numpy as np

# Defining data loader path
def load_data(path):
	df = pd.read_csv(path)
	return df


# Defining data cleaning function
def clean_data(df):
	
	# Dropping customer ID
	df = df.drop('customerID', axis = 1)

	# Converting TotalCharges to numeric
	df['TotalCharges'] = pd.to_numeric(df["TotalCharges"], errors= 'coerce')

	# Converting MonthlyCharges to numeriv
	df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors= 'coerce')

	# Handle missing values
	df['MonthlyCharges'].fillna(df['MonthlyCharges'].median(), inplace= True)
	df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace= True)

	df.dropna(inplace= True)
	
	# Converting SeniorCitizen column to str
	df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

	# Returning cleaned dataset
	return df
