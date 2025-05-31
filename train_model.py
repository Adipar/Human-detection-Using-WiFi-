import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("C:/Users/aadip/Downloads/wifi_rssi_modified.csv")

# Ensure 'Human_Present' column exists
if "Human_Present" not in df.columns:
    raise ValueError("Label column 'Human_Present' is missing. Manually label your data before training.")

X = df.drop(columns=["Timestamp", "Human_Present"]).values  # RSSI values
y = df["Human_Present"].values  # Labels (0 or 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for inference
np.save("scaler.npy", scaler.mean_)

# Build neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Output probability (0-1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Save model
model.save("wifi_human_detection_model.h5")
print("Model training complete and saved.")
