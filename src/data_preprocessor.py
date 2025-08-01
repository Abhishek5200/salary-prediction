import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class SalaryDataPreprocessor:
    """
    Data preprocessing pipeline for salary prediction
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.preprocessor = None
        self.feature_names = None
        self.target_column = 'salary'
        
    def create_preprocessing_pipeline(self, X):
        """
        Create preprocessing pipeline for features
        """
        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical features: {categorical_features}")
        print(f"Numerical features: {numerical_features}")
        
        # Create transformers
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def fit_transform(self, data):
        """
        Fit the preprocessor and transform the data
        """
        # Separate features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        # Create and fit preprocessing pipeline
        self.create_preprocessing_pipeline(X)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        self.feature_names = self.get_feature_names()
        
        print(f"Original features: {X.shape[1]}")
        print(f"Processed features: {X_processed.shape[1]}")
        
        return X_processed, y
    
    def transform(self, data):
        """
        Transform new data using fitted preprocessor
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        if self.target_column in data.columns:
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]
            X_processed = self.preprocessor.transform(X)
            return X_processed, y
        else:
            X_processed = self.preprocessor.transform(data)
            return X_processed
    
    def get_feature_names(self):
        """
        Get feature names after preprocessing
        """
        feature_names = []
        
        # Get numerical feature names
        num_features = self.preprocessor.named_transformers_['num']
        if hasattr(num_features, 'get_feature_names_out'):
            num_names = num_features.get_feature_names_out()
        else:
            # Fallback for older sklearn versions
            num_selector = self.preprocessor.named_transformers_['num']
            num_names = [f"num__{i}" for i in range(len(num_selector.named_steps))]
        
        # Get categorical feature names
        cat_features = self.preprocessor.named_transformers_['cat']
        if hasattr(cat_features, 'get_feature_names_out'):
            cat_names = cat_features.get_feature_names_out()
        else:
            # Fallback for older sklearn versions
            cat_names = [f"cat__{i}" for i in range(10)]  # Approximate
        
        feature_names.extend(num_names)
        feature_names.extend(cat_names)
        
        return feature_names
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """
        Save the fitted preprocessor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """
        Load a fitted preprocessor
        """
        saved_data = joblib.load(filepath)
        self.preprocessor = saved_data['preprocessor']
        self.feature_names = saved_data['feature_names']
        print(f"Preprocessor loaded from {filepath}")
    
    def get_data_stats(self, data):
        """
        Get basic statistics about the data
        """
        stats = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'salary_stats': data[self.target_column].describe().to_dict() if self.target_column in data.columns else None
        }
        return stats

def load_and_preprocess_data():
    """
    Load and preprocess the salary data
    """
    # Load training data
    train_data = pd.read_csv('data/salary_train.csv')
    test_data = pd.read_csv('data/salary_test.csv')
    
    print("Data loaded successfully!")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize preprocessor
    preprocessor = SalaryDataPreprocessor()
    
    # Get data statistics
    train_stats = preprocessor.get_data_stats(train_data)
    print("\nTraining data statistics:")
    print(f"Shape: {train_stats['shape']}")
    print(f"Missing values: {train_stats['missing_values']}")
    
    # Preprocess training data
    X_train, y_train = preprocessor.fit_transform(train_data)
    
    # Preprocess test data
    X_test, y_test = preprocessor.transform(test_data)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Split training data into train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\nData preprocessing completed!")
    print(f"Training set: {X_train_split.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return {
        'X_train': X_train_split,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train_split,
        'y_val': y_val,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names
    }

if __name__ == "__main__":
    # Generate sample data first
    from data.sample_salary_data import save_sample_data
    save_sample_data()
    
    # Preprocess the data
    processed_data = load_and_preprocess_data()
    print("Data preprocessing pipeline executed successfully!")