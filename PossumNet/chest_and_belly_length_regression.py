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

df = df.drop(labels = ["case", "site", "Pop"], axis = 1) # Remove these columns from the dataset since they don't help the model learn better

# Encode non-numeric values
df.sex = df.sex.map({'f': 1, 'm': 0})

# Fill in missing values
df['age'] = df['age'].fillna(df.age.mean())
df['footlgth'] = df['footlgth'].fillna(df.footlgth.mean())

# Scale x values
scaler = StandardScaler()
for col in df.columns:
  if col != 'chest' and col != 'belly':
    df[col] = scaler.fit_transform(df[[col]])

# Initialize x and y lists
x = []
y = []
    
# Add specific parts of the dataset to x and y lists
for row in range(df.shape[0]):
  rows = []
  for point in range(len(df.loc[0]) - 2): # "- 2" because we don't want to add the last two columns (the labels) to the inputs section
    rows.append(df.iloc[row][point])
  x.append(rows)
  y.append([df.loc[row][-1], df.loc[row][-2]]) # Add belly length and chest length to y-list since those are the labels

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

# Output layer
model.add(Dense(2)) # 2 neurons because there are 2 outputs (chest length and belly length)

# Compile model
model.compile(optimizer = 'sgd', loss = 'mse') # Mean squared error loss function and stochastic gradient descent optimizer since this is a regression model
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Train model and store training history
epochs = 200
history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_val, y_val)) # To add callbacks add 'callbacks = [early_stopping]'

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = model.predict([x_test[index]])
print(f"\nModel's Prediction on a Sample Input: {prediction}")
print(f"Actual Label on the Same Input: {y_test[index]}")

# Visualize  loss and validation loss
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
error1 = []
error2 = []

for val in range(len(x_test)): # Loop through test values and have model predict on those test values
  # Determine the difference between the model's predicted labels and actual labels
  error_val1 = abs(model.predict([x_test[val]])[0][0] - y_test[val][0])
  error_val2 = abs(model.predict([x_test[val]])[0][1] - y_test[val][1])

  # Store difference values in a list for plotting
  error1.append(float(error_val1))
  error2.append(float(error_val2))


# Create bar graph to illustrate the deviation between the model's predicted values and actual values
index = np.arange(len(error1))
bar_width = 0.35

bar1 = plt.bar(index, error1, bar_width, label = 'Chest Length Deviation')
bar2 = plt.bar(index + bar_width, error2, bar_width, color = 'orange', label = 'Belly Length Deviation')

plt.xlabel('Input Index')
plt.ylabel('Deviation')
plt.title('Approximate Model Error')
plt.legend()

plt.tight_layout()
plt.show()
