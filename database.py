import sqlite3
import datetime

def init_db():
    conn = sqlite3.connect('fault_detection.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            temperature REAL,
            humidity REAL,
            oil_level REAL,
            gas_value REAL,
            vibration INTEGER,
            prediction INTEGER,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect('fault_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

def add_user(username, password):
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                    (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return user

def add_prediction(user_id, temperature, humidity, oil_level, gas_value, vibration, prediction, confidence):
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO predictions (user_id, temperature, humidity, oil_level, gas_value, vibration, prediction, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, temperature, humidity, oil_level, gas_value, vibration, prediction, confidence))
    conn.commit()
    conn.close()

def get_user_predictions(user_id):
    conn = get_db_connection()
    predictions = conn.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (user_id,)).fetchall()
    conn.close()
    return predictions