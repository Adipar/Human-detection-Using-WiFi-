import numpy as np
import tensorflow as tf
import time
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load trained model and scaler
model = tf.keras.models.load_model("C:/Users/aadip/PycharmProjects/wifi_detection/.venv/Scripts/wifi_human_detection_model.h5")
scaler_mean = np.load("scaler.npy")

# Define number of WiFi APs (Assuming they are positioned in 3D space)
NUM_APS = 5
AP_POSITIONS = np.array([
    [0, 0, 2],  # AP 1 (Corner 1)
    [5, 0, 2],  # AP 2 (Corner 2)
    [0, 5, 2],  # AP 3 (Corner 3)
    [5, 5, 2],  # AP 4 (Corner 4)
    [2.5, 2.5, 3]  # AP 5 (Ceiling)
])

# Define laptop position (Assuming it's near the center of the room at table height)
LAPTOP_POSITION = np.array([2.5, 2.5, 1])

# Function to simulate RSSI readings
def get_rssi():
    return [random.randint(-80, -40) for _ in range(NUM_APS)]

# Function to estimate human position using weighted averaging
def estimate_position(rssi_values):
    weights = np.exp(np.array(rssi_values) / 10)  # Convert RSSI to weight
    position = np.average(AP_POSITIONS, axis=0, weights=weights)
    return position

# Start video capture
cap = cv2.VideoCapture(0)

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print("Starting real-time human detection... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get RSSI values and predict
    rssi_values = np.array(get_rssi()).reshape(1, -1) - scaler_mean
    prediction = model.predict(rssi_values)[0][0]

    # Estimate position of human
    human_pos = estimate_position(rssi_values[0])

    # Overlay detection result on frame
    if prediction > 0.8:  # Adjust threshold if needed
        text = "🚨 Human Detected!"
        color = (0, 0, 255)  # Red
    else:
        text = "No Human Detected"
        color = (0, 255, 0)  # Green

    # Draw rectangle and text
    cv2.rectangle(frame, (50, 50), (400, 100), color, -1)
    cv2.putText(frame, text, (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("WiFi-based Human Detection", frame)

    # Update 3D plot
    ax.clear()
    ax.scatter(AP_POSITIONS[:, 0], AP_POSITIONS[:, 1], AP_POSITIONS[:, 2], c='blue', marker='^', s=100, label="WiFi APs")
    ax.scatter(LAPTOP_POSITION[0], LAPTOP_POSITION[1], LAPTOP_POSITION[2], c='green', marker='s', s=150, label="Laptop")

    if prediction > 0.8:
        ax.scatter(human_pos[0], human_pos[1], human_pos[2], c='red', s=100, label="Human Detected")
    elif prediction > 0.5:
        ax.scatter(human_pos[0], human_pos[1], human_pos[2], c='yellow', s=100, label="Possible Human")

    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 3])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Height')
    ax.set_title("Real-Time Human Detection with WiFi")
    ax.legend()
    plt.pause(0.1)  # Update the plot

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.close()
print("Real-time detection stopped.")
