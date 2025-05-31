import numpy as np
import pandas as pd
import time
import random
from datetime import datetime

# Define WiFi Access Points (APs)
NUM_APS = 5  # Adjust based on your environment


# Function to get RSSI values (simulate actual readings)
def get_rssi():
    return [random.randint(-80, -40) for _ in range(NUM_APS)]  # Simulated RSSI values


# Create CSV file if not exists
csv_filename = "wifi_rssi_raw_data.csv"
try:
    df = pd.read_csv(csv_filename)
    print(f"Appending data to {csv_filename}")
except FileNotFoundError:
    print(f"Creating new file: {csv_filename}")
    columns = [f"AP{i + 1}" for i in range(NUM_APS)] + ["Timestamp"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_filename, index=False)

print("Starting continuous WiFi RSSI data collection... Press Ctrl+C to stop.")

try:
    while True:
        rssi_values = get_rssi()  # Get RSSI readings
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get timestamp

        # Save data
        new_data = rssi_values + [timestamp]
        df.loc[len(df)] = new_data
        df.to_csv(csv_filename, mode='a', header=False, index=False)

        print(f"Logged RSSI at {timestamp}: {rssi_values}")
        time.sleep(1)  # Adjust collection interval

except KeyboardInterrupt:
    print("\nData collection stopped.")
