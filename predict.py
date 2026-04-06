import numpy as np
import joblib
import tensorflow as tf
import requests

# Paths
MODEL_PATH = "results/lstm_model.h5"
SCALER_PATH = "results/scaler.pkl"

# Load model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Threshold values for maintenance prediction
THRESHOLDS = {
    'temperature': {'normal': (20, 35), 'warning': (35, 45), 'critical': (45, float('inf'))},
    'humidity': {'normal': (30, 60), 'warning': (60, 80), 'critical': (80, float('inf'))},
    'oil_level': {'normal': (15, float('inf')), 'warning': (10, 15), 'critical': (0, 10)},
    'gas_value': {'normal': (0, 500), 'warning': (500, 800), 'critical': (800, float('inf'))}
}

def get_realtime_data():
    """Fetch real-time data from ThingSpeak"""
    try:
        data = requests.get("https://api.thingspeak.com/channels/3107386/feeds.json?api_key=OMZ04XD69ZF78KAU&results=2")
        data_json = data.json()
        latest_feed = data_json['feeds'][-1]
        
        return {
            'temperature': float(latest_feed['field1']),
            'humidity': float(latest_feed['field2']),
            'oil_level': float(latest_feed['field3']),
            'gas_value': float(latest_feed['field4']),
            'vibration': int(latest_feed['field5'])
        }
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return default values if API fails
        return {
            'temperature': 24.34,
            'humidity': 47.65,
            'oil_level': 10.87,
            'gas_value': 594.91,
            'vibration': 0
        }

def predict_fault(temperature, humidity, oil_level, gas_value):
    """Predict fault class from input parameters"""
    single_input = np.array([[temperature, humidity, oil_level, gas_value]])
    
    # Scale & reshape
    scaled_input = scaler.transform(single_input)
    scaled_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))
    
    # Predict
    pred = model.predict(scaled_input)
    fault_class = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred))
    
    return fault_class, confidence

def get_fault_label(fault_class):
    """Convert fault class number to label"""
    fault_labels = {
        0: "Normal",
        1: "High Heat",
        2: "High Humidity", 
        3: "Low Oil Level",
        4: "Gas Leak"
    }
    return fault_labels.get(fault_class, "Unknown")

def calculate_maintenance_days(temperature, humidity, oil_level, gas_value, vibration):
    """Calculate maintenance days based on input parameters"""
    maintenance_info = {
        'temperature': {'days': None, 'status': 'normal', 'message': ''},
        'humidity': {'days': None, 'status': 'normal', 'message': ''},
        'oil_level': {'days': None, 'status': 'normal', 'message': ''},
        'gas_value': {'days': None, 'status': 'normal', 'message': ''},
        'vibration': {'days': None, 'status': 'normal', 'message': ''}
    }
    
    # Temperature analysis
    if temperature > THRESHOLDS['temperature']['critical'][0]:
        maintenance_info['temperature']['days'] = 1
        maintenance_info['temperature']['status'] = 'critical'
        maintenance_info['temperature']['message'] = f'Critical: Temperature {temperature}°C is too high'
    elif temperature > THRESHOLDS['temperature']['warning'][0]:
        maintenance_info['temperature']['days'] = 7
        maintenance_info['temperature']['status'] = 'warning'
        maintenance_info['temperature']['message'] = f'Warning: Temperature {temperature}°C is elevated'
    else:
        maintenance_info['temperature']['days'] = 30
        maintenance_info['temperature']['status'] = 'normal'
        maintenance_info['temperature']['message'] = f'Normal: Temperature {temperature}°C is within range'
    
    # Humidity analysis
    if humidity > THRESHOLDS['humidity']['critical'][0]:
        maintenance_info['humidity']['days'] = 2
        maintenance_info['humidity']['status'] = 'critical'
        maintenance_info['humidity']['message'] = f'Critical: Humidity {humidity}% is too high'
    elif humidity > THRESHOLDS['humidity']['warning'][0]:
        maintenance_info['humidity']['days'] = 10
        maintenance_info['humidity']['status'] = 'warning'
        maintenance_info['humidity']['message'] = f'Warning: Humidity {humidity}% is elevated'
    else:
        maintenance_info['humidity']['days'] = 30
        maintenance_info['humidity']['status'] = 'normal'
        maintenance_info['humidity']['message'] = f'Normal: Humidity {humidity}% is within range'
    
    # Oil level analysis
    if oil_level < THRESHOLDS['oil_level']['critical'][1]:
        maintenance_info['oil_level']['days'] = 1
        maintenance_info['oil_level']['status'] = 'critical'
        maintenance_info['oil_level']['message'] = f'Critical: Oil level {oil_level} is too low'
    elif oil_level < THRESHOLDS['oil_level']['warning'][1]:
        maintenance_info['oil_level']['days'] = 3
        maintenance_info['oil_level']['status'] = 'warning'
        maintenance_info['oil_level']['message'] = f'Warning: Oil level {oil_level} is low'
    else:
        maintenance_info['oil_level']['days'] = 30
        maintenance_info['oil_level']['status'] = 'normal'
        maintenance_info['oil_level']['message'] = f'Normal: Oil level {oil_level} is adequate'
    
    # Gas value analysis
    if gas_value > THRESHOLDS['gas_value']['critical'][0]:
        maintenance_info['gas_value']['days'] = 1
        maintenance_info['gas_value']['status'] = 'critical'
        maintenance_info['gas_value']['message'] = f'Critical: Gas value {gas_value} indicates leak'
    elif gas_value > THRESHOLDS['gas_value']['warning'][0]:
        maintenance_info['gas_value']['days'] = 5
        maintenance_info['gas_value']['status'] = 'warning'
        maintenance_info['gas_value']['message'] = f'Warning: Gas value {gas_value} is elevated'
    else:
        maintenance_info['gas_value']['days'] = 30
        maintenance_info['gas_value']['status'] = 'normal'
        maintenance_info['gas_value']['message'] = f'Normal: Gas value {gas_value} is safe'
    
    # Vibration analysis
    if vibration == 1:
        maintenance_info['vibration']['days'] = 2
        maintenance_info['vibration']['status'] = 'critical'
        maintenance_info['vibration']['message'] = 'Critical: High vibration detected'
    else:
        maintenance_info['vibration']['days'] = 30
        maintenance_info['vibration']['status'] = 'normal'
        maintenance_info['vibration']['message'] = 'Normal: Vibration levels are safe'
    
    # Calculate overall maintenance days (take the minimum)
    all_days = [info['days'] for info in maintenance_info.values() if info['days'] is not None]
    overall_days = min(all_days) if all_days else 30
    
    return maintenance_info, overall_days

def get_maintenance_recommendation(fault_class, maintenance_days):
    """Get maintenance recommendation based on fault class and days"""
    recommendations = {
        0: f"Routine maintenance recommended in {maintenance_days} days",
        1: f"Cooling system maintenance required in {maintenance_days} days",
        2: f"Humidity control maintenance needed in {maintenance_days} days",
        3: f"Oil system maintenance critical in {maintenance_days} days",
        4: f"Gas leak inspection required immediately in {maintenance_days} days"
    }
    return recommendations.get(fault_class, f"Maintenance recommended in {maintenance_days} days")