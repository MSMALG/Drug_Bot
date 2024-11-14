import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("processed_bioactivity_data.csv")

#Features to use (excluding the target variable)
X = df[['MoleculeWeight', 'LogP', 'NumHDonors', 'NumHAcceptors', 'pIC50']]

#Target variable (bioactivity classification: active vs inactive)
le = LabelEncoder()
df['bioactivity'] = le.fit_transform(df['bioactivity'])

y = df['bioactivity']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ExtraTreesRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model 
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

joblib.dump(model, 'extra_trees_model.pkl')
print('Model saved as extra_trees_model.pkl')

joblib.dump(le, 'label_encoder.pkl')
print('Label Encoder saved as label_encoder.pkl')
