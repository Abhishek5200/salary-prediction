from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ibm_watson_integration import SalaryPredictionService
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False
    print("Watson integration not available. Using local models only.")

app = Flask(__name__)
CORS(app)

# Global variables
prediction_service = None
local_model = None
preprocessor = None

def load_local_model():
    """Load local model if Watson is not available"""
    global local_model, preprocessor
    try:
        local_model = joblib.load('../models/best_model.pkl')
        preprocessor_data = joblib.load('../models/preprocessor.pkl')
        preprocessor = preprocessor_data['preprocessor']
        print("Local model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading local model: {e}")
        return False

def initialize_app():
    """Initialize the application"""
    global prediction_service
    
    if WATSON_AVAILABLE:
        try:
            prediction_service = SalaryPredictionService()
            # Note: In production, you'd want to setup Watson service here
            # prediction_service.setup_service()
            print("Watson service initialized")
        except Exception as e:
            print(f"Watson service initialization failed: {e}")
            load_local_model()
    else:
        load_local_model()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get form data
        data = request.json if request.is_json else request.form
        
        job_title = data.get('job_title', 'Software Engineer')
        years_experience = int(data.get('years_experience', 5))
        education_level = data.get('education_level', 'Bachelor')
        location = data.get('location', 'New York')
        company_size = data.get('company_size', 'Medium')
        industry = data.get('industry', 'Technology')
        performance_rating = float(data.get('performance_rating', 3.5))
        certifications = int(data.get('certifications', 1))
        
        # Make prediction
        prediction_result = make_prediction(
            job_title, years_experience, education_level, location,
            company_size, industry, performance_rating, certifications
        )
        
        if request.is_json:
            return jsonify(prediction_result)
        else:
            return render_template('result.html', result=prediction_result)
            
    except Exception as e:
        error_result = {
            'status': 'error',
            'error': str(e),
            'predicted_salary': None
        }
        
        if request.is_json:
            return jsonify(error_result), 400
        else:
            return render_template('result.html', result=error_result)

def make_prediction(job_title, years_experience, education_level, location,
                   company_size, industry, performance_rating, certifications):
    """Make salary prediction using available model"""
    
    try:
        # Try Watson service first
        if prediction_service and WATSON_AVAILABLE:
            result = prediction_service.predict_salary_from_features(
                job_title, years_experience, education_level, location,
                company_size, industry, performance_rating, certifications
            )
            result['method'] = 'Watson ML'
            return result
        
        # Fallback to local model
        elif local_model and preprocessor:
            # Create feature DataFrame
            features_df = pd.DataFrame({
                'job_title': [job_title],
                'years_experience': [years_experience],
                'education_level': [education_level],
                'location': [location],
                'company_size': [company_size],
                'industry': [industry],
                'performance_rating': [performance_rating],
                'certifications': [certifications]
            })
            
            # Preprocess features
            features_processed = preprocessor.transform(features_df)
            
            # Make prediction
            predicted_salary = local_model.predict(features_processed)[0]
            
            return {
                'predicted_salary': round(predicted_salary, 2),
                'status': 'success',
                'method': 'Local Model',
                'features': features_df.iloc[0].to_dict()
            }
        
        else:
            return {
                'status': 'error',
                'error': 'No model available for prediction',
                'predicted_salary': None
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'predicted_salary': None
        }

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    return predict()

@app.route('/chat')
def chat():
    """Chat interface page"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat API endpoint"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if prediction_service and WATSON_AVAILABLE:
            result = prediction_service.chat_predict_salary(message)
            return jsonify(result)
        else:
            # Simple fallback response
            return jsonify({
                'response': 'Chat functionality requires Watson Assistant. Please use the prediction form instead.',
                'type': 'error'
            })
            
    except Exception as e:
        return jsonify({
            'response': f'Error: {str(e)}',
            'type': 'error'
        }), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'watson_available': WATSON_AVAILABLE,
        'local_model_available': local_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize the application
    initialize_app()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)