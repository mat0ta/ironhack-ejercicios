# We are going to create a model and train it

# Import the libraries
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

# Load the data
print(pathlib.Path(__file__).parent.parent)
project_path = pathlib.Path(__file__).parent.parent
df = pd.read_csv('./creditcardmarketing.csv')

# Split the data into X and y
X = df.drop('Offer_Accepted', axis = True)
y = df['Offer_Accepted']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
y_pred = log.predict(X_test)
score_train = log.score(X_train, y_train)
score_test = log.score(X_test, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
recall_test = recall_score(y_test, y_pred)
recall_train = recall_score(y_train, log.predict(X_train))
f1 = f1_score(y_test, y_pred)
f1_train = f1_score(y_train, log.predict(X_train))
tt_data = [score_train, score_test, precision, recall, recall_train, recall_test, f1, f1_train]

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(classification_report(y_test, y_pred))

# Save the model
pickle.dump(log, open('./creditcardmarketing.pkl', 'wb'))
