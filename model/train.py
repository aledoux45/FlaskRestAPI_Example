# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle

seed = 42

# Load data
# X, y = load_iris(return_X_y=True, as_frame=False)
data = pd.read_csv("data/iris.csv")
X = data.iloc[:,:4].values
y = data.iloc[:,-1].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Scale variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
model = LogisticRegression(random_state=seed, verbose=0)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Output performance 
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", cm)

# Save model & standard scaler
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(sc, open('model/scaler.pkl', 'wb'))

print("Done training!")