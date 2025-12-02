import flask
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Model ---
try:
    # IMPORTANT: The model file 'michelin_model.pkl' must be in the same directory
    # as this app.py file, and it must have been created by running train_model.py.
    with open('michelin_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model 'michelin_model.pkl' loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model file 'michelin_model.pkl' not found.")
    model = None # Set model to None to handle errors later
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

# Define the expected feature columns based on a typical restaurant dataset
# These must match the columns the model was trained on, excluding the target.
# Inferring from typical use case for a Michelin Star prediction model:
REQUIRED_COLUMNS = ['Cuisine', 'Price_Range', 'Location', 'Review_Score']
CUISINE_OPTIONS = ['French', 'Italian', 'Japanese', 'American', 'Other']
PRICE_OPTIONS = ['Low', 'Medium', 'High', 'Very High']
LOCATION_OPTIONS = ['Major City', 'Rural Area', 'Suburban']
STAR_LABELS = {0: '0 Stars', 1: '1 Star', 2: '2 Stars', 3: '3 Stars'}

# Function to safely make a prediction
def make_prediction(data):
    """Takes a single dictionary of features and returns the prediction."""
    if model is None:
        return "Model not available due to load error.", "error"

    try:
        # Create a DataFrame from the input data
        # Ensure the order and format match the training data
        input_df = pd.DataFrame([data], columns=REQUIRED_COLUMNS)

        # Make prediction
        prediction_array = model.predict(input_df)
        prediction_prob = model.predict_proba(input_df)

        # Get the predicted class (0, 1, 2, or 3 stars)
        predicted_star_index = prediction_array[0]
        predicted_star_label = STAR_LABELS.get(predicted_star_index, 'Unknown')

        # Get the confidence for the predicted class
        confidence = prediction_prob[0][predicted_star_index] * 100

        result = {
            'prediction': predicted_star_label,
            'confidence': f"{confidence:.2f}%",
            'raw_index': int(predicted_star_index)
        }
        return result, "success"

    except ValueError as ve:
        # This often happens if the input features are unexpected (e.g., missing columns)
        error_msg = f"Prediction Error: Missing or incorrect features. Expected: {REQUIRED_COLUMNS}. Details: {ve}"
        print(error_msg)
        return error_msg, "error"
    except Exception as e:
        error_msg = f"An unexpected error occurred during prediction: {e}"
        print(error_msg)
        return error_msg, "error"


# --- Routes ---

@app.route('/')
def home():
    """Renders the main prediction page."""
    # Pass the options to the HTML template for dynamic form creation
    options = {
        'cuisines': CUISINE_OPTIONS,
        'prices': PRICE_OPTIONS,
        'locations': LOCATION_OPTIONS
    }
    return render_template('index.html', options=options)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and returns the prediction as JSON."""
    try:
        # Get data from the form submission
        data = request.form.to_dict()

        # Convert the Review_Score to a float/int
        if 'Review_Score' in data and data['Review_Score']:
            data['Review_Score'] = float(data['Review_Score'])
        else:
            # Handle case where Review_Score is missing/empty
            return jsonify({'status': 'error', 'message': 'Review Score is required.'}), 400

        # Create a dictionary suitable for the model
        input_data = {
            'Cuisine': data.get('cuisine'),
            'Price_Range': data.get('price_range'),
            'Location': data.get('location'),
            'Review_Score': data.get('Review_Score')
        }

        # Make the prediction
        result, status = make_prediction(input_data)

        if status == "success":
            return jsonify({
                'status': 'success',
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'raw_index': result['raw_index'],
                'input': input_data # Echo back the input for review
            })
        else:
            return jsonify({'status': 'error', 'message': result}), 500

    except Exception as e:
        print(f"Server-side error: {e}")
        return jsonify({'status': 'error', 'message': f'Internal Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the application
    # In a production environment, use a more robust server like Gunicorn
    app.run(debug=True)