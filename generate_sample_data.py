import pandas as pd
import numpy as np

# Generate sample data for testing
def generate_sample_data():
    # Create a dictionary with sample data
    data = {
        'ID': range(1, 101),
        'Name': [f'Person {i}' for i in range(1, 101)],
        'Age': np.random.randint(18, 65, 100),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], 100),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 100),
        'Salary': np.random.randint(30000, 120000, 100),
        'Department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'Operations'], 100),
        'Years_Experience': np.random.randint(0, 30, 100),
        'Performance_Score': np.random.uniform(1, 5, 100).round(2),
        'Hire_Date': pd.date_range(start='2010-01-01', periods=100, freq='M')
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values
    for col in ['Age', 'Salary', 'Years_Experience', 'Performance_Score']:
        mask = np.random.choice([True, False], size=df.shape[0], p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    # Add some duplicates
    duplicate_indices = np.random.choice(df.index, size=5, replace=False)
    for idx in duplicate_indices:
        df = pd.concat([df, df.iloc[[idx]]], ignore_index=True)
    
    return df

# Generate sample data
sample_df = generate_sample_data()

# Save to different formats for testing
sample_df.to_csv('/home/ubuntu/data_analytics_app/sample_data.csv', index=False)
sample_df.to_excel('/home/ubuntu/data_analytics_app/sample_data.xlsx', index=False)
sample_df.to_json('/home/ubuntu/data_analytics_app/sample_data.json', orient='records')

# Create a SQLite database with the sample data
import sqlite3
conn = sqlite3.connect('/home/ubuntu/data_analytics_app/sample_data.db')
sample_df.to_sql('employees', conn, if_exists='replace', index=False)
conn.close()

print("Sample data files created successfully!")
