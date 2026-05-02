import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data():
    """
    Complete pipeline for the Auto MPG Dataset.
    Includes acquisition, cleaning, feature engineering, encoding, and scaling.
    """
    print("Starting Data Pipeline...")

    # --- 1. DATA ACQUISITION ---
    # Fetching directly from the UCI Machine Learning Repository
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                    'Acceleration', 'Model Year', 'Origin']
    
    # Reading whitespace-separated values; '?' is treated as NaN
    df = pd.read_csv(url, names=column_names, na_values='?', 
                     comment='\t', sep=' ', skipinitialspace=True)

    # --- 2. DATA CLEANING ---
    # Filling the 6 missing Horsepower values with the median (Better than dropping)
    df['Horsepower'] = df['Horsepower'].fillna(df['Horsepower'].median())

    # --- 3. FEATURE ENGINEERING ---
    # Creating 'Power_to_Weight' ratio - a key predictor for fuel efficiency
    df['Power_to_Weight'] = df['Horsepower'] / df['Weight']

    # --- 4. ONE-HOT ENCODING ---
    # Mapping numeric origin to labels and then creating dummy variables
    df['Origin'] = df['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    df = pd.get_dummies(df, columns=['Origin'], prefix='', prefix_sep='')

    # --- 5. TRAIN/TEST SPLIT ---
    # Defining features (X) and target variable (y)
    X = df.drop('MPG', axis=1)
    y = df['MPG']

    # 80/20 split with a fixed random_state for consistency across the group
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 6. NORMALIZATION (SCALING) ---
    # Standardizing features to have mean=0 and variance=1
    scaler = StandardScaler()
    
    # We fit the scaler ONLY on the training data to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Returning everything needed for the Analyst and ML Engineer
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, df

if __name__ == "__main__":
    # Execute the pipeline
    X_train, X_test, y_train, y_test, feature_names, full_df = prepare_data()
    
    # --- 7. EXPORT TO EXCEL ---
    # Reconstructing DataFrames for the Excel file
    train_export = pd.DataFrame(X_train, columns=feature_names)
    train_export['MPG_Actual'] = y_train.values
    
    test_export = pd.DataFrame(X_test, columns=feature_names)
    test_export['MPG_Actual'] = y_test.values

    file_name = 'Cleaned_Car_Data.xlsx'
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        train_export.to_excel(writer, sheet_name='Training_Data', index=False)
        test_export.to_excel(writer, sheet_name='Testing_Data', index=False)
        # Also saving the raw cleaned version for the Analyst
        full_df.to_excel(writer, sheet_name='Full_Cleaned_Dataset', index=False)

    print("-" * 30)
    print(f"SUCCESS: Pipeline finished.")
    print(f"File Saved: {file_name}")
    print(f"Features Engineered: {list(feature_names)}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")
    print("-" * 30)