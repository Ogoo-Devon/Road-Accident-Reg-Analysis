# INSTALLATION AND SETUPS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#DATA PEPROCSSING
from google.colab import files
uploaded = files.upload()

raw_data = pd.read_csv("RTA Dataset.csv")
data = raw_data.copy()
data.head()

data.isnull().sum()

data.dropna(inplace=True)

data.drop(["Educational_level", "Work_of_casuality"], axis = 1, inplace = True)

X = data.drop("Accident_severity", axis = 1)
y = data["Accident_severity"]

# VARIABLES ENCODING

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

y_Encode = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_Encode, test_size = 0.2, random_state = 42)

#Train a Random Forest Classifier (Scikit-learn)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# Predict and evaluate the model
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Feature importance from Random Forest
feature_importance = rf_model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance from Random Forest')
plt.show()


#ADDING A CONSTANT
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()
model.summary()

y_pred = model.predict(X_test)

#CONVERTING PEDICTION TO BINARY

y_pred_binary = (y_pred > 0.5).astype(int)
print(y_pred_binary)

#MODEL EVALUATION
print("Accuracy Score: ", accuracy_score(y_test, y_pred_binary))
print("Classification Report: ")
print(classification_report(y_test, y_pred_binary))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_binary))

plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(len(y_test)), y=y_pred_binary)
plt.title('Predicted Probabilities vs Actual Outcome ')
plt.xlabel('Predicted Probabilities')
plt.ylabel('Actual Outcome')
plt.show()

