# predict_single_input.py
import numpy as np
import joblib
import tensorflow as tf

# Paths
MODEL_PATH = "results/lstm_model.h5"
SCALER_PATH = "results/scaler.pkl"

# Load model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Example single input [temp, humidity, oil_level, gas_value]
import requests
data=requests.get("https://api.thingspeak.com/channels/3107386/feeds.json?api_key=OMZ04XD69ZF78KAU&results=2")
temp=data.json()['feeds'][-1]['field1']
hum=data.json()['feeds'][-1]['field2']
oil=data.json()['feeds'][-1]['field3']
gas=data.json()['feeds'][-1]['field4']
vibration=data.json()['feeds'][-1]['field5']
#24.34,47.65,10.87,594.91
single_input = np.array([[temp,hum,oil,gas]])

if vibration == 1:
    print("high vibration detected")
else:
    print("no vibration")

# Scale & reshape
scaled_input = scaler.transform(single_input)
scaled_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))

# Predict
pred = model.predict(scaled_input)
fault_class = np.argmax(pred, axis=1)[0]
print(f"Predicted fault class: {fault_class}")
