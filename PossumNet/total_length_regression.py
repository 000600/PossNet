# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Get input shape
input_shape = len(x[0])

# Create model
model = Sequential()

# Add an initial batch norm layer so that all the values are in a reasonable range for the network to process
model.add(BatchNormalization())
model.add(Dense(9, activation = 'relu', input_shape = [input_shape])) # Input layer

# Hidden layers
model.add(Dense(5, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))

# Output layer
model.add(Dense(1)) # 1 neuron because the model predicts 1 class (total length)

# Compile model
model.compile(optimizer = 'sgd', loss = 'mae') # Mean absolute error loss function and stochastic gradient descent optimizer since this is a regression model
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Train model and store training history
epochs = 200
history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_val, y_val)) # To add callbacks add 'callbacks = [early_stopping]'

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = model.predict([x_test[index]])[0]
print(f"\nModel's Prediction on a Sample Input: {prediction}")
print(f"Actual Label on the Same Input: {y_test[index]}")

# Visualize loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epoch_list = [i for i in range(epochs)]

plt.plot(epoch_list, loss, label = 'Loss')
plt.plot(epoch_list, val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

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
plt.title('Model Error')

plt.show()
