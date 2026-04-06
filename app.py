from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import database as db
import predict
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production

# Initialize database
db.init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = db.get_user(username)
        if user and user['password'] == password:  # In production, use proper password hashing
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
        elif len(username) < 3:
            flash('Username must be at least 3 characters long', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
        else:
            if db.add_user(username, password):
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Username already exists', 'error')
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get prediction history
    predictions = db.get_user_predictions(session['user_id'])
    return render_template('dashboard.html', predictions=predictions)

# @app.route('/predict', methods=['GET', 'POST'])
# @login_required
# def predict_page():
#     # Get real-time data
#     realtime_data = predict.get_realtime_data()
    
#     prediction_result = None
#     confidence = None
    
#     if request.method == 'POST':
#         # Get form data
#         temperature = float(request.form['temperature'])
#         humidity = float(request.form['humidity'])
#         oil_level = float(request.form['oil_level'])
#         gas_value = float(request.form['gas_value'])
#         vibration = int(request.form.get('vibration', 0))
#         inps=[temperature, humidity, oil_level, gas_value, vibration]
        
#         # Make prediction
#         fault_class, confidence = predict.predict_fault(temperature, humidity, oil_level, gas_value)
#         fault_label = predict.get_fault_label(fault_class)

#         print(f"\n\n\n\n\n Output is {fault_label} \n\n\n\n\n")

        
        
#         # Save to database
#         db.add_prediction(session['user_id'], temperature, humidity, oil_level, gas_value, vibration, int(fault_class), confidence)
#         if vibration == 1:
#             vib_status="HIGH VIBRATION"
#         else:
#             vib_status=None
#         prediction_result = {
#             'fault_class': fault_class,
#             'fault_label': fault_label,
#             'confidence': confidence
#         }
    
#     return render_template('predict.html', 
#                          data=realtime_data, 
#                          prediction=prediction_result,vib_status=vib_status,inps=inps)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict_page():
    # Get real-time data
    realtime_data = predict.get_realtime_data()
    
    prediction_result = None
    confidence = None
    maintenance_info = None
    overall_days = None
    maintenance_recommendation = None
    vib_status = None
    inps = None
    
    if request.method == 'POST':
        # Get form data
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        oil_level = float(request.form['oil_level'])
        gas_value = float(request.form['gas_value'])
        vibration = int(request.form.get('vibration', 0))
        inps = [temperature, humidity, oil_level, gas_value, vibration]
        
        # Make prediction
        fault_class, confidence = predict.predict_fault(temperature, humidity, oil_level, gas_value)
        fault_label = predict.get_fault_label(fault_class)

        print(f"\n\n\n\n\n Output is {fault_label} \n\n\n\n\n")
        
        # Calculate maintenance information
        maintenance_info, overall_days = predict.calculate_maintenance_days(
            temperature, humidity, oil_level, gas_value, vibration
        )
        
        # Get maintenance recommendation
        maintenance_recommendation = predict.get_maintenance_recommendation(fault_class, overall_days)
        
        # Save to database
        db.add_prediction(session['user_id'], temperature, humidity, oil_level, gas_value, vibration, int(fault_class), confidence)
        
        if vibration == 1:
            vib_status = "HIGH VIBRATION"
        else:
            vib_status = None
            
        prediction_result = {
            'fault_class': fault_class,
            'fault_label': fault_label,
            'confidence': confidence
        }
    
    return render_template('predict.html', 
                         data=realtime_data, 
                         prediction=prediction_result,
                         vib_status=vib_status,
                         inps=inps,
                         maintenance_info=maintenance_info,
                         overall_days=overall_days,
                         maintenance_recommendation=maintenance_recommendation)

@app.route('/get_realtime_data')
@login_required
def get_realtime_data():
    """API endpoint to get real-time data (for AJAX updates)"""
    data = predict.get_realtime_data()
    return jsonify(data)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)