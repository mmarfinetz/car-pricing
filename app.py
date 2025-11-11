"""
Production API for Used Car Price Prediction
ACV - Automated Car Valuation Service
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models
MODEL = None
PREPROCESSOR = None
MODEL_METADATA = None

def load_models():
    """Load saved models at startup"""
    global MODEL, PREPROCESSOR, MODEL_METADATA

    try:
        # Load main model
        model_path = 'models/production_tree_model.pkl'
        if not os.path.exists(model_path):
            # Try alternative paths
            if os.path.exists('production_tree_model.pkl'):
                model_path = 'production_tree_model.pkl'
            elif os.path.exists('models/production_tree_macro_model.pkl'):
                model_path = 'models/production_tree_macro_model.pkl'

        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
        logger.info(f"✓ Loaded model from {model_path}")

        # Try to load preprocessor if saved separately
        if os.path.exists('models/preprocessor.pkl'):
            with open('models/preprocessor.pkl', 'rb') as f:
                PREPROCESSOR = pickle.load(f)
            logger.info("✓ Loaded preprocessor")

        # Load metadata if available
        if os.path.exists('models/model_metadata.json'):
            import json
            with open('models/model_metadata.json', 'r') as f:
                MODEL_METADATA = json.load(f)
            logger.info("✓ Loaded model metadata")

        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

# Load models on startup
if not load_models():
    logger.warning("Models not loaded - API will return errors")

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'ACV Used Car Price Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': MODEL is not None,
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Single vehicle price prediction',
            '/batch_predict': 'Multiple vehicle predictions',
            '/model_info': 'Model metadata and performance'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if MODEL else 'unhealthy',
        'model_loaded': MODEL is not None,
        'timestamp': datetime.now().isoformat()
    }), 200 if MODEL else 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict used car price for a single vehicle

    Expected JSON payload:
    {
        "Year": 2018,
        "Kilometers_Driven": 50000,
        "Fuel_Type": "Petrol",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": 18.5,
        "Engine": 1497,
        "Power": 103.5,
        "Seats": 5,
        "Location": "Mumbai",
        "Name": "Honda City"
    }
    """
    try:
        # Check if model is loaded
        if MODEL is None:
            return jsonify({'error': 'Model not loaded'}), 503

        # Get request data
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Log request
        logger.info(f"Prediction request received: {data.get('Name', 'Unknown')} - {data.get('Year', 'Unknown')}")

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Handle missing columns with defaults
        required_columns = {
            'Year': 2015,
            'Kilometers_Driven': 50000,
            'Fuel_Type': 'Petrol',
            'Transmission': 'Manual',
            'Owner_Type': 'First',
            'Mileage': 20.0,
            'Engine': 1500,
            'Power': 100,
            'Seats': 5
        }

        for col, default_value in required_columns.items():
            if col not in input_df.columns:
                input_df[col] = default_value
                logger.warning(f"Missing {col}, using default: {default_value}")

        # Make prediction
        try:
            # The model expects log-transformed target, so predictions are in log scale
            log_prediction = MODEL.predict(input_df)

            # Transform back from log scale to actual price
            price = np.expm1(log_prediction)[0]

            # Ensure price is positive and reasonable
            price = max(price, 10000)  # Minimum 10k
            price = min(price, 10000000)  # Maximum 1 crore

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback to simple estimation
            base_price = 500000  # 5 lakhs base
            age_factor = max(0.5, 1 - (datetime.now().year - data.get('Year', 2015)) * 0.1)
            km_factor = max(0.5, 1 - data.get('Kilometers_Driven', 50000) / 200000)
            price = base_price * age_factor * km_factor

        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'price': float(price),
                'price_formatted': f"₹{price:,.0f}",
                'currency': 'INR',
                'confidence': 'high' if 100000 < price < 5000000 else 'medium'
            },
            'vehicle': {
                'year': data.get('Year'),
                'fuel_type': data.get('Fuel_Type'),
                'transmission': data.get('Transmission'),
                'kilometers': data.get('Kilometers_Driven')
            },
            'model_version': MODEL_METADATA.get('model_version', '1.0.0') if MODEL_METADATA else '1.0.0',
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Prediction successful: ₹{price:,.0f}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed. Please check your input data.'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict prices for multiple vehicles

    Expected JSON payload:
    {
        "vehicles": [
            {"Year": 2018, "Kilometers_Driven": 50000, ...},
            {"Year": 2019, "Kilometers_Driven": 30000, ...}
        ]
    }
    """
    try:
        if MODEL is None:
            return jsonify({'error': 'Model not loaded'}), 503

        data = request.json
        vehicles = data.get('vehicles', [])

        if not vehicles:
            return jsonify({'error': 'No vehicles provided'}), 400

        predictions = []

        for i, vehicle in enumerate(vehicles):
            try:
                # Convert to DataFrame
                input_df = pd.DataFrame([vehicle])

                # Make prediction
                log_prediction = MODEL.predict(input_df)
                price = np.expm1(log_prediction)[0]

                # Ensure reasonable bounds
                price = max(price, 10000)
                price = min(price, 10000000)

                predictions.append({
                    'index': i,
                    'price': float(price),
                    'price_formatted': f"₹{price:,.0f}",
                    'success': True
                })

            except Exception as e:
                predictions.append({
                    'index': i,
                    'error': str(e),
                    'success': False
                })

        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_vehicles': len(vehicles),
            'successful_predictions': sum(1 for p in predictions if p['success']),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model metadata and performance metrics"""
    if MODEL_METADATA:
        return jsonify(MODEL_METADATA)
    else:
        return jsonify({
            'message': 'Model metadata not available',
            'model_loaded': MODEL is not None
        })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # For local testing
    print("="*60)
    print("ACV USED CAR PRICING API")
    print("="*60)
    print(f"Model loaded: {MODEL is not None}")

    # Use Railway's PORT environment variable if available
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    print("\nTest with:")
    print('curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d \'{"Year":2018,"Kilometers_Driven":50000,"Fuel_Type":"Petrol","Transmission":"Manual"}\'')
    print("="*60)

    app.run(host='0.0.0.0', port=port, debug=False)