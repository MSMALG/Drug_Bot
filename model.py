import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

# Load the processed dataset
df = pd.read_csv("processed_bioactivity_data.csv")

# Features to use (excluding the target variable)
X = df[['MoleculeWeight', 'LogP', 'NumHDonors', 'NumHAcceptors', 'pIC50']]

# Target variable (bioactivity classification: active vs inactive)
# We'll need to encode the bioactivity class into numeric values (active=1, inactive=0)
le = LabelEncoder()
df['bioactivity'] = le.fit_transform(df['bioactivity'])

y = df['bioactivity']  # Target variable (encoded as 1 for active, 0 for inactive)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the ExtraTreesRegressor model
model = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model (you can use accuracy, RMSE, or any other metric)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model using joblib
joblib.dump(model, 'extra_trees_model.pkl')
print('Model saved as extra_trees_model.pkl')

# Optionally, save the label encoder for future use (to reverse the encoding of predictions)
joblib.dump(le, 'label_encoder.pkl')
print('Label Encoder saved as label_encoder.pkl')
