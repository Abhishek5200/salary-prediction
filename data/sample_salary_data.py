import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import os

def generate_sample_salary_data(n_samples=10000, random_state=42):
    """
    Generate sample salary data for the ML model training
    """
    np.random.seed(random_state)
    
    # Generate base features
    X, y_base = make_regression(n_samples=n_samples, n_features=8, noise=0.1, random_state=random_state)
    
    # Create realistic feature names and data
    job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'Marketing Manager', 
                  'Sales Representative', 'HR Manager', 'Finance Analyst', 'Operations Manager',
                  'DevOps Engineer', 'UX Designer', 'Business Analyst', 'Project Manager']
    
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    locations = ['New York', 'San Francisco', 'Austin', 'Seattle', 'Chicago', 'Boston', 
                'Los Angeles', 'Denver', 'Atlanta', 'Remote']
    
    data = {
        'job_title': np.random.choice(job_titles, n_samples),
        'years_experience': np.random.exponential(5, n_samples).astype(int),
        'education_level': np.random.choice(education_levels, n_samples),
        'location': np.random.choice(locations, n_samples),
        'company_size': np.random.choice(['Startup', 'Small', 'Medium', 'Large'], n_samples),
        'industry': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'], n_samples),
        'performance_rating': np.random.uniform(1, 5, n_samples),
        'certifications': np.random.randint(0, 6, n_samples),
    }
    
    # Create salary based on realistic factors
    base_salary = 50000
    df = pd.DataFrame(data)
    
    # Apply salary multipliers based on features
    salary_multipliers = {
        'job_title': {
            'Software Engineer': 1.3, 'Data Scientist': 1.4, 'Product Manager': 1.35,
            'Marketing Manager': 1.1, 'Sales Representative': 0.9, 'HR Manager': 1.0,
            'Finance Analyst': 1.05, 'Operations Manager': 1.15, 'DevOps Engineer': 1.35,
            'UX Designer': 1.2, 'Business Analyst': 1.1, 'Project Manager': 1.25
        },
        'education_level': {
            'High School': 0.8, 'Bachelor': 1.0, 'Master': 1.2, 'PhD': 1.4
        },
        'location': {
            'San Francisco': 1.5, 'New York': 1.4, 'Seattle': 1.3, 'Boston': 1.2,
            'Los Angeles': 1.15, 'Chicago': 1.05, 'Austin': 1.1, 'Denver': 1.0,
            'Atlanta': 0.95, 'Remote': 1.1
        },
        'company_size': {
            'Startup': 0.9, 'Small': 1.0, 'Medium': 1.1, 'Large': 1.3
        }
    }
    
    salary = base_salary
    for feature, multipliers in salary_multipliers.items():
        salary *= df[feature].map(multipliers)
    
    # Add experience and performance factors
    salary *= (1 + df['years_experience'] * 0.03)  # 3% per year of experience
    salary *= df['performance_rating'] / 3  # Performance impact
    salary *= (1 + df['certifications'] * 0.02)  # 2% per certification
    
    # Add some noise
    salary *= np.random.normal(1, 0.1, n_samples)
    
    # Ensure minimum salary
    salary = np.maximum(salary, 30000)
    
    df['salary'] = salary.astype(int)
    
    return df

def save_sample_data():
    """Save sample data to CSV file"""
    os.makedirs('data', exist_ok=True)
    
    # Generate and save training data
    train_data = generate_sample_salary_data(8000, random_state=42)
    train_data.to_csv('data/salary_train.csv', index=False)
    
    # Generate and save test data
    test_data = generate_sample_salary_data(2000, random_state=123)
    test_data.to_csv('data/salary_test.csv', index=False)
    
    print("Sample salary data generated and saved:")
    print(f"Training data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    print("\nTraining data preview:")
    print(train_data.head())
    print(f"\nSalary statistics:")
    print(train_data['salary'].describe())

if __name__ == "__main__":
    save_sample_data()