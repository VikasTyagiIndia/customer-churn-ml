# Importing libraries
from sklearn.metrics import classification_report, roc_auc_score

# Defining evaluation function
def evaluate(model, X_test, y_test):
	# Getting predicted values
	y_pred = model.predict(X_test)

	# Getting prediction probabilities
	y_proba = model.predict_proba(X_test)[:, 1]

	# Getting classification report
	print("="*100)
	print(" "*30,"CLASSIFICATION REPORT")
	print("="*100)

	print(classification_report(y_test, y_pred))

	# Getting ROC-AUC Score
	print("="*100)
	print(" "*30 ,"ROC-AUC SCORE: ",roc_auc_score(y_test, y_proba))
	print("="*100)