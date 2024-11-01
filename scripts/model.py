import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load the preprocessed data
data_path = 'C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/data/preprocessed_stock_data.csv'
data = pd.read_csv(data_path)

# Use only 'Close' prices for now
close_data = data[['Date', 'Symbol', 'Close']]

# Filter data for a single symbol (e.g., AAPL)
close_data = close_data[close_data['Symbol'] == 'AAPL']
close_data = close_data[['Date', 'Close']]
close_data.set_index('Date', inplace=True)

# Convert to numpy array
close_prices = close_data.values

# Feature scaling with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Prepare the data for LSTM (look-back window of 60 days)
def prepare_data(sequence, look_back=60):
    X, y = [], []
    for i in range(look_back, len(sequence)):
        X.append(sequence[i-look_back:i, 0])
        y.append(sequence[i, 0])
    return np.array(X), np.array(y)

# Prepare training and testing datasets
look_back = 60
X, y = prepare_data(scaled_data, look_back)

# Ensure X is not empty before reshaping
if X.shape[0] == 0:
    raise ValueError("Not enough data to create training samples. Please ensure you have sufficient data.")

# Reshape X to be [samples, time steps, features]
print(f"Shape of X before reshaping: {X.shape}")
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into training and testing data
split_ratio = 0.8
split = int(len(X) * split_ratio)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Inverse scale the predictions and actual values
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten()))
print(f'Root Mean Squared Error: {rmse}')

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test_rescaled, color='blue', label='Actual Stock Price')
plt.plot(y_pred_rescaled, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction (Baseline LSTM Model)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()