import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class SalaryEnsembleModel:
    """
    Ensemble learning model for salary prediction
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.best_model = None
        self.model_performance = {}
        
    def create_base_models(self):
        """
        Create base models for ensemble
        """
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'svr': SVR(kernel='rbf', C=100, gamma='scale')
        }
        
        print("Base models created:")
        for name in self.models.keys():
            print(f"- {name}")
    
    def create_ensemble_model(self):
        """
        Create voting regressor ensemble
        """
        self.ensemble_model = VotingRegressor([
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boosting']),
            ('lr', self.models['linear_regression'])
        ])
        print("Ensemble model (Voting Regressor) created")
    
    def train_individual_models(self, X_train, y_train, X_val, y_val):
        """
        Train individual models and evaluate performance
        """
        print("Training individual models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            train_r2 = r2_score(y_train, y_pred_train)
            val_r2 = r2_score(y_val, y_pred_val)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            
            # Store performance
            self.model_performance[name] = {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'val_mae': val_mae
            }
            
            print(f"  Train RMSE: ${train_rmse:,.2f}")
            print(f"  Val RMSE: ${val_rmse:,.2f}")
            print(f"  Val R²: {val_r2:.4f}")
            print(f"  Val MAE: ${val_mae:,.2f}")
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """
        Train ensemble model
        """
        print("\nTraining ensemble model...")
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.ensemble_model.predict(X_train)
        y_pred_val = self.ensemble_model.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        
        # Store performance
        self.model_performance['ensemble'] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'val_mae': val_mae
        }
        
        print(f"Ensemble Model Performance:")
        print(f"  Train RMSE: ${train_rmse:,.2f}")
        print(f"  Val RMSE: ${val_rmse:,.2f}")
        print(f"  Val R²: {val_r2:.4f}")
        print(f"  Val MAE: ${val_mae:,.2f}")
    
    def find_best_model(self):
        """
        Find the best performing model based on validation R²
        """
        best_r2 = -np.inf
        best_model_name = None
        
        for name, metrics in self.model_performance.items():
            if metrics['val_r2'] > best_r2:
                best_r2 = metrics['val_r2']
                best_model_name = name
        
        if best_model_name == 'ensemble':
            self.best_model = self.ensemble_model
        else:
            self.best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name} (Val R²: {best_r2:.4f})")
        return best_model_name, self.best_model
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """
        Perform hyperparameter tuning for a specific model
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5, 10]
            }
            model = GradientBoostingRegressor(random_state=42)
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}:")
        print(grid_search.best_params_)
        print(f"Best cross-validation score: {-grid_search.best_score_:.2f}")
        
        return grid_search.best_estimator_
    
    def evaluate_on_test_set(self, X_test, y_test):
        """
        Evaluate the best model on test set
        """
        print("\nEvaluating best model on test set...")
        
        y_pred_test = self.best_model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"Test Set Performance:")
        print(f"  RMSE: ${test_rmse:,.2f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAE: ${test_mae:,.2f}")
        
        return {
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'predictions': y_pred_test
        }
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the best model (if available)
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            if feature_names is not None:
                feature_importance = pd.DataFrame({
                    'feature': feature_names[:len(importance)],
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Feature Importances:")
                print(feature_importance.head(10))
                
                return feature_importance
            else:
                return importance
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_models(self, model_dir='models'):
        """
        Save all trained models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            filepath = os.path.join(model_dir, f'{name}_model.pkl')
            joblib.dump(model, filepath)
        
        # Save ensemble model
        if self.ensemble_model is not None:
            ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
            joblib.dump(self.ensemble_model, ensemble_path)
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = os.path.join(model_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
        
        # Save performance metrics
        performance_path = os.path.join(model_dir, 'model_performance.pkl')
        joblib.dump(self.model_performance, performance_path)
        
        print(f"Models saved to {model_dir} directory")
    
    def load_best_model(self, model_path='models/best_model.pkl'):
        """
        Load the best model
        """
        self.best_model = joblib.load(model_path)
        print(f"Best model loaded from {model_path}")
    
    def predict_salary(self, features):
        """
        Predict salary for new features
        """
        if self.best_model is None:
            raise ValueError("No model trained. Train a model first or load a saved model.")
        
        prediction = self.best_model.predict(features.reshape(1, -1))
        return prediction[0]
    
    def create_performance_summary(self):
        """
        Create a summary of model performance
        """
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        df_performance = pd.DataFrame(self.model_performance).T
        df_performance = df_performance.round(4)
        
        print(df_performance)
        
        return df_performance

def train_ensemble_models():
    """
    Main function to train ensemble models
    """
    # Import and load preprocessed data
    from data_preprocessor import load_and_preprocess_data
    
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    
    # Initialize ensemble model
    ensemble_model = SalaryEnsembleModel()
    
    # Create and train models
    ensemble_model.create_base_models()
    ensemble_model.create_ensemble_model()
    
    # Train individual models
    ensemble_model.train_individual_models(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Train ensemble model
    ensemble_model.train_ensemble_model(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Find best model
    best_model_name, best_model = ensemble_model.find_best_model()
    
    # Evaluate on test set
    test_results = ensemble_model.evaluate_on_test_set(
        data['X_test'], data['y_test']
    )
    
    # Get feature importance
    feature_importance = ensemble_model.get_feature_importance(data['feature_names'])
    
    # Save models
    ensemble_model.save_models()
    
    # Create performance summary
    performance_summary = ensemble_model.create_performance_summary()
    
    print(f"\nModel training completed successfully!")
    print(f"Best model: {best_model_name}")
    print(f"Test R²: {test_results['test_r2']:.4f}")
    
    return ensemble_model, test_results, feature_importance

if __name__ == "__main__":
    ensemble_model, test_results, feature_importance = train_ensemble_models()