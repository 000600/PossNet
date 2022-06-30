# Imports
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('possum_data.csv')
df = pd.DataFrame(df)

# Map non numeric values
df.sex = df.sex.map({'f': 1, 'm': 0})

# Fill in missing values
df['age'] = df['age'].fillna(df.age.mean())
df['footlgth'] = df['footlgth'].fillna(df.footlgth.mean())

# Remove unnecessary columns from the dataset
df = df.drop(labels = ["case", "site", "Pop"], axis = 1) # These columns will not assist the model in learning how to predict the total length of a possum

# Scale x values
scaler = StandardScaler()
for col in df.columns:
  if col != 'totlngth':
    df[col] = scaler.fit_transform(df[[col]])

# Initialize x and y lists
x = []
y = list(df.pop("totlngth"))
    
# Add dataset to x and y lists
for row in range(df.shape[0]):
  rows = []
  for point in range(len(df.loc[0])): # Loop through all columns
    rows.append(df.iloc[row][point])
  x.append(rows)

# Divide the x and y values into three sets: train, test, and validation
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5, random_state = 1)

# Create and train model
model = XGBRegressor(n_estimators = 50000, learning_rate = 0.001)
model.fit(x_train, y_train, early_stopping_rounds = 5, eval_set = [(x_val, y_val)], verbose = 1) # Predicts the total length of a possum

# View mean squared error of the model
predictions = model.predict(x_test)
mse = mean_squared_error(predictions, y_test)
print("\nMean Squared Error (MSE):", mse)

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = model.predict([x_test[index]])[0]
print(f"Model's Prediction on a Sample Input: {prediction}")
print(f"Actual Label on the Same Input: {y_test[index]}")

# Calculate model's approximate deviation
error = []
for val in range(len(x_test)): # Loop through test values and have model predict on those test values
  error_val = abs(model.predict([x_test[val]]) - y_test[val])[0] # Determine the difference between the model's predicted labels and actual labels
  error.append(float(error_val)) # Store difference values in a list for plotting

# Visualize deviation
y_pos = np.arange(len(error))

plt.figure(figsize = (8, 6))
plt.bar(y_pos, error, align = 'center')
plt.ylabel('Deviation')
plt.xlabel('Input Index')
plt.title('XGBoost Error')

plt.show()
