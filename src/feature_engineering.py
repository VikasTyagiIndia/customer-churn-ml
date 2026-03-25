# Importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_features(df):
	"""
	Create new features to improve model.

	WHY:

	Raw data rarely contains directly useful signals.
	Feature engineering extracts hidden patterns -> bigger AUC boost.
	"""

	df = df.copy()

	# Group tenure into categories
	# WHY:
	# Models learn better from grouped behavior patterns than raw numbers

	df['tenure_group'] = pd.cut(
            df['tenure'],
            bins = [0,12,24,48,72],
            labels = ['0-1 yr', '1-2 yr', '2-4 yr', '4-6 yr'])

	# Average Monthly Proxy
	# WHY:
	# Capture spending behavior more effectively than raw totals

	df['avg_charge'] = df['TotalCharges'] / (df['tenure'] + 1)
	
	# Count no. of services used
	# WHY:
	# Customers using more services are less likely to churn
	
	services = [
	"PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
	"DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
	]
	
	# Count how many services = 'Yes'
	df["num_services"] = (df[services] == 'Yes').sum(axis= 1)
	
	return df
	
def encode_features(df):
	"""
	Convert categorical features into numerical format.
	
	WHY:
	ML models cannot process text directly, need numeric encoding.
	"""

	# One hot encoding
	# WHY:
	# Avoids introducing artificial order
	df = pd.get_dummies(df, drop_first = True)

	return df

def split_data(df):
	"""
	Seperate features(X) and target(y)

	WHY:
	Standard ML workflow requires clear seperation of input and output.
	"""

	X = df.drop("Churn_Yes", axis= 1)
	y = df['Churn_Yes']

	return X,y
