# Human Detection Using WiFi

This project leverages WiFi signals to detect human presence by analyzing Received Signal Strength Indicator (RSSI) values. It offers a privacy-preserving and cost-effective solution for indoor human detection without relying on cameras or additional sensors.

## 🧠 Overview

The system operates by collecting RSSI data from nearby WiFi networks, training a machine learning model to recognize patterns indicative of human presence, and performing real-time detection based on live RSSI readings.

## 📁 Repository Structure

- `collect_data.py`: Script to scan available WiFi networks and record RSSI values, generating a dataset for model training.
- `wifi_rssi_raw_data.csv`: Sample dataset containing collected RSSI readings.
- `train_model.py`: Script to train a machine learning model using the collected RSSI data.
- `wifi_human_detection_model.h5`: Pre-trained model file for human detection based on RSSI inputs.
- `realtime_detection.py`: Script to perform real-time human detection using live RSSI data and the trained model.

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Required Python libraries:

```bash
pip install numpy pandas scikit-learn tensorflow
