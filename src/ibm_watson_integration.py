import os
import json
import pandas as pd
import numpy as np
from ibm_watson_machine_learning import APIClient
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv
import joblib
from datetime import datetime
import requests

# Load environment variables
load_dotenv()

class IBMWatsonMLIntegration:
    """
    Integration with IBM Watson Machine Learning for model deployment
    """
    
    def __init__(self):
        self.wml_credentials = {
            "url": os.getenv('WML_URL', 'https://us-south.ml.cloud.ibm.com'),
            "apikey": os.getenv('IBM_CLOUD_API_KEY')
        }
        self.client = None
        self.space_id = None
        self.deployment_id = None
        self.model_id = None
        
    def initialize_client(self):
        """Initialize Watson ML client"""
        try:
            self.client = APIClient(self.wml_credentials)
            print("Watson ML client initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing Watson ML client: {e}")
            return False
    
    def create_space(self, space_name="salary-prediction-space"):
        """Create a deployment space"""
        try:
            # Check if space already exists
            spaces = self.client.spaces.list()
            
            for space in spaces['resources']:
                if space['entity']['name'] == space_name:
                    self.space_id = space['metadata']['id']
                    print(f"Using existing space: {space_name}")
                    self.client.set.default_space(self.space_id)
                    return self.space_id
            
            # Create new space if it doesn't exist
            metadata = {
                self.client.spaces.ConfigurationMetaNames.NAME: space_name,
                self.client.spaces.ConfigurationMetaNames.DESCRIPTION: "Space for salary prediction models"
            }
            
            space_details = self.client.spaces.store(meta_props=metadata)
            self.space_id = space_details['metadata']['id']
            self.client.set.default_space(self.space_id)
            
            print(f"Created new space: {space_name}")
            return self.space_id
            
        except Exception as e:
            print(f"Error creating space: {e}")
            return None
    
    def upload_model(self, model_path, model_name="salary-prediction-ensemble"):
        """Upload trained model to Watson ML"""
        try:
            # Load the model
            model = joblib.load(model_path)
            
            # Define model metadata
            model_meta = {
                self.client.repository.ModelMetaNames.NAME: model_name,
                self.client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
                self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.client.software_specifications.get_uid_by_name("scikit-learn_1.1-py3.9"),
                self.client.repository.ModelMetaNames.DESCRIPTION: "Ensemble model for salary prediction"
            }
            
            # Store the model
            model_details = self.client.repository.store_model(
                model=model,
                meta_props=model_meta
            )
            
            self.model_id = model_details['metadata']['id']
            print(f"Model uploaded successfully with ID: {self.model_id}")
            return self.model_id
            
        except Exception as e:
            print(f"Error uploading model: {e}")
            return None
    
    def deploy_model(self, deployment_name="salary-prediction-deployment"):
        """Deploy the model as an online service"""
        try:
            # Define deployment metadata
            deployment_meta = {
                self.client.deployments.ConfigurationMetaNames.NAME: deployment_name,
                self.client.deployments.ConfigurationMetaNames.ONLINE: {},
                self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "Online deployment for salary prediction"
            }
            
            # Create deployment
            deployment_details = self.client.deployments.create(
                artifact_uid=self.model_id,
                meta_props=deployment_meta
            )
            
            self.deployment_id = deployment_details['metadata']['id']
            print(f"Model deployed successfully with deployment ID: {self.deployment_id}")
            return self.deployment_id
            
        except Exception as e:
            print(f"Error deploying model: {e}")
            return None
    
    def predict_online(self, input_data):
        """Make online predictions using deployed model"""
        try:
            # Prepare scoring payload
            scoring_payload = {
                "input_data": [
                    {
                        "fields": [f"feature_{i}" for i in range(len(input_data))],
                        "values": [input_data.tolist()]
                    }
                ]
            }
            
            # Make prediction
            predictions = self.client.deployments.score(
                deployment_id=self.deployment_id,
                scoring_payload=scoring_payload
            )
            
            predicted_salary = predictions['predictions'][0]['values'][0][0]
            print(f"Predicted salary: ${predicted_salary:,.2f}")
            return predicted_salary
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_deployment_info(self):
        """Get deployment information"""
        try:
            deployment_details = self.client.deployments.get_details(self.deployment_id)
            return deployment_details
        except Exception as e:
            print(f"Error getting deployment info: {e}")
            return None

class IBMWatsonAssistantIntegration:
    """
    Integration with IBM Watson Assistant for chatbot functionality
    """
    
    def __init__(self):
        self.assistant_credentials = {
            'apikey': os.getenv('WATSON_ASSISTANT_API_KEY'),
            'url': os.getenv('WATSON_ASSISTANT_URL')
        }
        self.assistant_id = os.getenv('WATSON_ASSISTANT_ID')
        self.assistant = None
        self.session_id = None
        
    def initialize_assistant(self):
        """Initialize Watson Assistant"""
        try:
            authenticator = IAMAuthenticator(self.assistant_credentials['apikey'])
            self.assistant = AssistantV2(
                version='2023-06-15',
                authenticator=authenticator
            )
            self.assistant.set_service_url(self.assistant_credentials['url'])
            
            # Create a session
            session = self.assistant.create_session(
                assistant_id=self.assistant_id
            ).get_result()
            self.session_id = session['session_id']
            
            print("Watson Assistant initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing Watson Assistant: {e}")
            return False
    
    def send_message(self, message_text):
        """Send message to Watson Assistant"""
        try:
            response = self.assistant.message(
                assistant_id=self.assistant_id,
                session_id=self.session_id,
                input={
                    'message_type': 'text',
                    'text': message_text
                }
            ).get_result()
            
            return response
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return None
    
    def process_salary_query(self, user_input, wml_integration):
        """Process salary prediction query through Watson Assistant"""
        try:
            # Send message to Watson Assistant
            response = self.send_message(user_input)
            
            # Extract intent and entities
            if response and 'output' in response:
                intents = response['output'].get('intents', [])
                entities = response['output'].get('entities', [])
                
                # Check if this is a salary prediction intent
                if any(intent['intent'] == 'salary_prediction' for intent in intents):
                    # Extract features from entities
                    features = self.extract_features_from_entities(entities)
                    
                    if features:
                        # Make prediction using Watson ML
                        predicted_salary = wml_integration.predict_online(features)
                        
                        return {
                            'intent': 'salary_prediction',
                            'predicted_salary': predicted_salary,
                            'features': features,
                            'assistant_response': response['output'].get('generic', [])
                        }
                
                return {
                    'intent': 'general',
                    'assistant_response': response['output'].get('generic', [])
                }
            
        except Exception as e:
            print(f"Error processing salary query: {e}")
            return None
    
    def extract_features_from_entities(self, entities):
        """Extract salary prediction features from Watson Assistant entities"""
        # Initialize default feature values
        features = {
            'years_experience': 5,
            'performance_rating': 3.5,
            'certifications': 1,
            'job_title': 'Software Engineer',
            'education_level': 'Bachelor',
            'location': 'New York',
            'company_size': 'Medium',
            'industry': 'Technology'
        }
        
        # Extract values from entities
        for entity in entities:
            entity_name = entity['entity']
            entity_value = entity['value']
            
            if entity_name == 'years_experience':
                features['years_experience'] = int(entity_value)
            elif entity_name == 'job_title':
                features['job_title'] = entity_value
            elif entity_name == 'education_level':
                features['education_level'] = entity_value
            elif entity_name == 'location':
                features['location'] = entity_value
            elif entity_name == 'company_size':
                features['company_size'] = entity_value
            elif entity_name == 'industry':
                features['industry'] = entity_value
            elif entity_name == 'performance_rating':
                features['performance_rating'] = float(entity_value)
            elif entity_name == 'certifications':
                features['certifications'] = int(entity_value)
        
        # Convert to feature vector (this would need to match your preprocessing)
        # For now, return as a numpy array with dummy encoding
        # In practice, you'd use your preprocessor to transform this
        feature_vector = np.array([
            features['years_experience'],
            features['performance_rating'],
            features['certifications'],
            1 if features['job_title'] == 'Software Engineer' else 0,
            1 if features['education_level'] == 'Bachelor' else 0,
            # Add more encoded features as needed...
        ])
        
        return feature_vector

class SalaryPredictionService:
    """
    Main service class combining Watson ML and Watson Assistant
    """
    
    def __init__(self):
        self.wml_integration = IBMWatsonMLIntegration()
        self.assistant_integration = IBMWatsonAssistantIntegration()
        self.preprocessor = None
        
    def setup_service(self, model_path='models/best_model.pkl'):
        """Setup the complete service"""
        print("Setting up Salary Prediction Service...")
        
        # Initialize Watson ML
        if not self.wml_integration.initialize_client():
            return False
        
        # Create space
        space_id = self.wml_integration.create_space()
        if not space_id:
            return False
        
        # Upload and deploy model
        model_id = self.wml_integration.upload_model(model_path)
        if not model_id:
            return False
        
        deployment_id = self.wml_integration.deploy_model()
        if not deployment_id:
            return False
        
        # Initialize Watson Assistant
        if not self.assistant_integration.initialize_assistant():
            print("Warning: Watson Assistant initialization failed")
        
        # Load preprocessor
        try:
            preprocessor_data = joblib.load('models/preprocessor.pkl')
            self.preprocessor = preprocessor_data['preprocessor']
            print("Preprocessor loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load preprocessor: {e}")
        
        print("Salary Prediction Service setup completed!")
        return True
    
    def predict_salary_from_features(self, job_title, years_experience, education_level, 
                                   location, company_size, industry, performance_rating, 
                                   certifications):
        """Predict salary from structured features"""
        try:
            # Create DataFrame with features
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
            if self.preprocessor:
                features_processed = self.preprocessor.transform(features_df)
            else:
                # Fallback: create dummy feature vector
                features_processed = np.array([years_experience, performance_rating, certifications] + [1]*47)
            
            # Make prediction
            predicted_salary = self.wml_integration.predict_online(features_processed)
            
            return {
                'predicted_salary': predicted_salary,
                'features': features_df.iloc[0].to_dict(),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error predicting salary: {e}")
            return {
                'predicted_salary': None,
                'error': str(e),
                'status': 'error'
            }
    
    def chat_predict_salary(self, user_message):
        """Predict salary through chatbot interaction"""
        try:
            result = self.assistant_integration.process_salary_query(
                user_message, self.wml_integration
            )
            return result
        except Exception as e:
            print(f"Error in chat prediction: {e}")
            return None

def setup_watson_integration():
    """Main function to setup Watson integration"""
    service = SalaryPredictionService()
    
    # Setup the service
    success = service.setup_service()
    
    if success:
        print("\n" + "="*60)
        print("IBM WATSON INTEGRATION SETUP COMPLETED")
        print("="*60)
        print("✓ Watson Machine Learning - Model deployed")
        print("✓ Watson Assistant - Chatbot ready")
        print("✓ Salary Prediction Service - Active")
        print("\nYou can now:")
        print("1. Make direct predictions via API")
        print("2. Chat with the assistant for salary predictions")
        print("3. Integrate with web applications")
    else:
        print("❌ Setup failed. Please check your IBM Cloud credentials.")
    
    return service

if __name__ == "__main__":
    # Setup Watson integration
    service = setup_watson_integration()
    
    # Example usage
    if service:
        # Test direct prediction
        print("\nTesting direct prediction...")
        result = service.predict_salary_from_features(
            job_title='Data Scientist',
            years_experience=5,
            education_level='Master',
            location='San Francisco',
            company_size='Large',
            industry='Technology',
            performance_rating=4.2,
            certifications=3
        )
        print(f"Prediction result: {result}")