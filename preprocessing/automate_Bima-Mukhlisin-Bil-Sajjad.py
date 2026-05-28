import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(raw_csv_path, output_csv_path):
    """
    Automated preprocessing pipeline for Cleveland Heart Disease Dataset.
    
    Steps:
    1. Load data
    2. Handle missing values (if any)
    3. Drop duplicates
    4. Outlier handling (clipping via IQR)
    5. Feature scaling (StandardScaler for continuous variables)
    6. Categorical encoding (One-Hot Encoding)
    7. Save preprocessed dataset
    """
    print(f"Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    
    # 1. Handle missing values
    # Numeric columns
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
    # Categorical columns
    cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            
    # 2. Drop duplicates
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    final_shape = df.shape
    print(f"Dropped {initial_shape[0] - final_shape[0]} duplicate rows. New shape: {df.shape}")
    
    # 3. Outlier handling: Clipping outliers via IQR for numeric columns
    # We clip values beyond [Q1 - 1.5*IQR, Q3 + 1.5*IQR] to the boundary values
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    print("Outliers clipped using IQR method.")
    
    # 4. Feature scaling
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("Continuous features standardized.")
    
    # 5. One-Hot Encoding for categorical features
    # Ensure categorical features are of category type first
    for col in cat_cols:
        df[col] = df[col].astype(str)
        
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Convert dummy columns (True/False) to 0/1 integers
    dummy_cols = [c for c in df_encoded.columns if c not in num_cols + ['sex', 'fbs', 'exang', 'target']]
    for c in dummy_cols:
        df_encoded[c] = df_encoded[c].astype(int)
        
    # Ensure binary fields are also integers
    for c in ['sex', 'fbs', 'exang', 'target']:
        if c in df_encoded.columns:
            df_encoded[c] = df_encoded[c].astype(int)
            
    # Save the preprocessed dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_encoded.to_csv(output_csv_path, index=False)
    print(f"Preprocessed dataset saved to {output_csv_path} (shape: {df_encoded.shape})")
    return df_encoded

if __name__ == '__main__':
    raw_path = r"c:\Users\bimam\Downloads\Eksperimen_SML_Bima Mukhlisin Bil Sajjad\heart_disease_raw\heart-disease.csv"
    output_path = r"c:\Users\bimam\Downloads\Eksperimen_SML_Bima Mukhlisin Bil Sajjad\preprocessing\heart_disease_preprocessing\heart-disease-preprocessed.csv"
    preprocess_data(raw_path, output_path)
