from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import os
from flask import send_from_directory
from tensorflow.keras.preprocessing import image as keras_image
import jsonify
from sqlalchemy.exc import IntegrityError
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
import secrets

secret_key = secrets.token_hex(16)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    dob = db.Column(db.Date, nullable=False)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String(100), nullable=False)
    result = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables within the application context
with app.app_context():
    db.create_all()

# Load the glaucoma detection model
model = load_model('combine_cnn6.h5')

# Define function for glaucoma prediction
def glaucoma_prediction(test_image_path):
    img = keras_image.load_img(test_image_path, target_size=(256, 256))
    image_array = img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    result = np.argmax(model.predict(image_array))
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']
        dob_str = request.form['dob']

        # Convert the dob string to a Python date object
        dob = datetime.strptime(dob_str, '%Y-%m-%d').date()

        # Check if the email already exists in the database
        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            # Return the registration form with an error message
            return render_template('register.html', error='Email address is already registered.')

        new_user = User(username=username, password=password, name=name, email=email, dob=dob)

        try:
            db.session.add(new_user)
            db.session.commit()

            # Add a print statement for debugging
            print(f"User {username} successfully registered.")
            
            return redirect(url_for('index'))
        except Exception as e:
            # Print the exception for debugging
            print(f"Error registering user: {str(e)}")

            # Return an error message to the user
            return render_template('register.html', error='An error occurred during registration.')

    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    print(f"Received login request - Username: {username}, Password: {password}")

    # Check if the user is an admin
    if username == 'admin' and password == 'admin123':
        session['user_id'] = 'admin'
        print("Redirecting to admin_dashboard")
        return redirect(url_for('admin_dashboard'))

    # If not admin, check regular user credentials
    user = User.query.filter_by(username=username, password=password).first()

    print(f"User: {user}")

    if user:
        session['user_id'] = user.id
        print("Redirecting to patient_dashboard")
        return redirect(url_for('patient_dashboard'))

    print("Invalid credentials or user not found")

    # Add these print statements for additional debugging
    print(f"Username: {username}, Password: {password}")
    all_users = User.query.all()
    print(f"All users in the database: {all_users}")

    return render_template('index.html', error='Invalid credentials or user not found')


@app.route('/view_users')
def view_users():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    # Check if the user is an admin
    if session.get('user_id') == 'admin':
        # Query all registered users from the database
        all_users = User.query.all()
        return render_template('view_users.html', all_users=all_users)
    else:
        return redirect(url_for('index'))

@app.route('/patient_dashboard')
def patient_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])
    return render_template('patient_page.html', patient_details=user)

from flask import jsonify  # Make sure to import jsonify

# ...

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        uploaded_file = request.files['file']

        # Save the uploaded image temporarily
        uploaded_image_path = os.path.join('static', 'uploads', uploaded_file.filename)
        uploaded_file.save(uploaded_image_path)

        # Make a prediction using the image classification model
        prediction = glaucoma_prediction(uploaded_image_path)

        # Determine the prediction result
        result = "Gluacoma" if prediction == 0 else "No Glaucoma"

        # Save the result to the database
        patient_id = session['user_id']
        new_result = Result(patient_id=patient_id, image_path=uploaded_image_path, result=result)
        db.session.add(new_result)
        db.session.commit()

        # Return the result and uploaded image URL
        return jsonify({'result': result, 'uploaded_image': uploaded_file.filename})

    except Exception as e:
        # Print the exception for debugging
        print(f"Error processing prediction: {str(e)}")

        return jsonify({'error': 'An error occurred while processing the prediction.'})


@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    # Check if the user is an admin
    if session.get('user_id') == 'admin':
        all_results = Result.query.all()
        return render_template('admin_page.html', all_results=all_results)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
