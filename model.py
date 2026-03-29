import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("data/health_data.csv")

le = LabelEncoder()
data['Smoking'] = le.fit_transform(data['Smoking'])
data['Alcohol'] = le.fit_transform(data['Alcohol'])
data['Risk'] = le.fit_transform(data['Risk'])

X = data.drop('Risk', axis=1)
y = data['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))