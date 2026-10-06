"""
Preprocessing shared by the modelling notebooks (``preprocess``) and the deployed
Gradio app (``transform_new_data``).

Both paths apply the same steps: median imputation, log transforms for tenure,
standard scaling, one-hot encoding and selection of the final 18 features.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ============================================================
# COLUMN DEFINITIONS
# ============================================================
# RAW columns to keep BEFORE any transformations
columns_to_keep = [
    # Target
    'Attrition',
    
    # Ordinal features (kept as-is)
    'JobSatisfaction', 'JobInvolvement', 'EnvironmentSatisfaction', 
    'WorkLifeBalance',
    
    # Numeric features (kept as-is)
    'NumCompaniesWorked',
    
    # For log transformation (only 2 features after testing)
    'TotalWorkingYears', 'YearsAtCompany',
    
    # Categorical columns (for one-hot encoding)
    'Department', 'EducationField', 'JobRole', 
    'BusinessTravel', 'MaritalStatus', 'OverTime'
]

# Define column types for processing
ordinal_cols = [
    'JobSatisfaction', 'JobInvolvement', 'EnvironmentSatisfaction', 'WorkLifeBalance'
]

numeric_cols = [
    'NumCompaniesWorked', 'TotalWorkingYears', 'YearsAtCompany'
]

# Only 2 log features after testing showed better Recall performance
log_features = [
    'TotalWorkingYearsLog', 'YearsAtCompanyLog'
    # REMOVED: 'MonthlyIncomeLog', 'YearsSinceLastPromotionLog' (worse model performance)
]

scale_cols = numeric_cols + ordinal_cols + log_features

drop_first_vars = ['BusinessTravel', 'Gender', 'MaritalStatus', 'OverTime']

# Final selected features (18 features after VIF analysis removed Job_happiness_score)
selected_features = [
    'JobSatisfaction', 'JobInvolvement', 'EnvironmentSatisfaction',
    'JobRole_Research Director', 'JobRole_Sales Representative', 
    'BusinessTravel_Travel_Frequently', 'WorkLifeBalance', 
    'Department_Research & Development', 'YearsAtCompanyLog',
    'TotalWorkingYearsLog', 'MaritalStatus_Single', 'OverTime_Yes', 
    'JobRole_Laboratory Technician', 'JobRole_Manufacturing Director', 
    'MaritalStatus_Married', 'JobRole_Manager', 'NumCompaniesWorked', 
    'JobRole_Healthcare Representative'
]
# REMOVED: 'Job_happiness_score' due to perfect multicollinearity


# ============================================================
# CUSTOM TRANSFORMERS
# ============================================================
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates log-transformed features (only 2 after testing)"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Only keep 2 log transformations that improved Recall
        for col, log_col in [
            ('TotalWorkingYears', 'TotalWorkingYearsLog'),
            ('YearsAtCompany', 'YearsAtCompanyLog')
            # REMOVED: MonthlyIncome, YearsSinceLastPromotion after testing
        ]:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors='coerce')
                series = series.fillna(series.median())
                series = series.clip(lower=0)
                X[log_col] = np.log1p(series)
            else:
                X[log_col] = np.nan
        return X


class CustomCategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encodes categorical variables"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Department - don't drop any
        if 'Department' in X.columns:
            dummies = pd.get_dummies(X['Department'], prefix='Department', drop_first=False)
            X = X.drop('Department', axis=1)
            X[dummies.columns] = dummies
        
        # EducationField - drop ONLY 'Other'
        if 'EducationField' in X.columns:
            edu_dummies = pd.get_dummies(X['EducationField'], prefix='EducationField')
            if 'EducationField_Other' in edu_dummies.columns:
                edu_dummies = edu_dummies.drop('EducationField_Other', axis=1)
            X = X.drop('EducationField', axis=1)
            X[edu_dummies.columns] = edu_dummies
        
        # JobRole - don't drop any
        if 'JobRole' in X.columns:
            jobrole_dummies = pd.get_dummies(X['JobRole'], prefix='JobRole', drop_first=False)
            X = X.drop('JobRole', axis=1)
            X[jobrole_dummies.columns] = jobrole_dummies
        
        # Drop-first for remaining categoricals
        for col in drop_first_vars:
            if col in X.columns:
                col_dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = X.drop(col, axis=1)
                X[col_dummies.columns] = col_dummies
        
        return X


class ColumnSynchronizer(BaseEstimator, TransformerMixin):
    """Ensures all datasets have the same columns"""
    def __init__(self, ref_columns):
        self.ref_columns = ref_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        missing_cols = set(self.ref_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[self.ref_columns].reset_index(drop=True)
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects final features"""
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        return X[self.features]


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================
def preprocess(data_path, target_name='Attrition', test_size=0.2, val_size=0.25, random_state=42):
    """
    Complete preprocessing pipeline - optimized for Recall performance
    
    Args:
        data_path: Path to raw CSV file
        target_name: Name of target variable
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test split)
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, fitted scaler, fitted imputer
    """
    
    # Load data and keep only specified columns
    df = pd.read_csv(data_path)
    
    # Keep only the columns we need
    available_cols = [col for col in columns_to_keep if col in df.columns]
    df = df[available_cols]
    
    print(f"Kept {len(available_cols)} columns from original dataset")
    
    # Encode target variable
    df[target_name] = df[target_name].map({'Yes': 1, 'No': 0})
    
    # Split target from features
    y = df[target_name]
    X = df.drop(columns=[target_name])
    
    # Train/Val/Test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=val_size, random_state=random_state
    )
    
    # --- Impute numeric columns (fit/train, transform all) and ensure numeric dtype
    num_imputer = SimpleImputer(strategy='median')
    for df_ in [X_train, X_val, X_test]:
        for col in numeric_cols:
            if col in df_.columns:
                df_[col] = pd.to_numeric(df_[col], errors='coerce')
    
    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = num_imputer.transform(X_val[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
    
    # --- Feature engineering (only 2 log features after testing)
    engineer = CustomFeatureEngineer()
    X_train = engineer.transform(X_train)
    X_val = engineer.transform(X_val)
    X_test = engineer.transform(X_test)
    
    # --- Scaling for numeric + ordinal + log columns
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    
    # --- Custom categorical encoding
    encoder = CustomCategoricalEncoder()
    X_train = encoder.transform(X_train)
    X_val = encoder.transform(X_val)
    X_test = encoder.transform(X_test)
    
    # --- Column synchronization
    all_columns = sorted(set(X_train.columns) | set(X_val.columns) | set(X_test.columns))
    synchronizer = ColumnSynchronizer(ref_columns=all_columns)
    X_train = synchronizer.transform(X_train)
    X_val = synchronizer.transform(X_val)
    X_test = synchronizer.transform(X_test)
    
    # --- Impute engineered features (robust forced fill, just in case)
    for df_ in [X_train, X_val, X_test]:
        for col in ['YearsAtCompanyLog', 'TotalWorkingYearsLog']:
            # REMOVED: 'MonthlyIncomeLog', 'YearsSinceLastPromotionLog', 'Job_happiness_score'
            if col in df_.columns:
                df_[col] = pd.to_numeric(df_[col], errors='coerce')
                df_[col] = df_[col].fillna(df_[col].median())
    
    # --- Feature selection (18 features after VIF analysis)
    selector = FeatureSelector(features=selected_features)
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)

    print("\nFinal feature shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"\nTotal features: {len(selected_features)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, num_imputer


# ============================================================
# TRANSFORM NEW DATA (FOR DEPLOYMENT)
# ============================================================

def transform_new_data(input_data, scaler, imputer):
    """
    Transform new employee data for prediction
    
    Args:
        input_data: dict or DataFrame with RAW employee features
        scaler: Pre-fitted StandardScaler from training
        imputer: Pre-fitted SimpleImputer from training
    
    Returns:
        DataFrame with processed features matching training data
    """
    # Convert to DataFrame if dict
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Keep only the columns we need (same as preprocess, but NO target)
    available_cols = [col for col in columns_to_keep if col in df.columns and col != 'Attrition']
    df = df[available_cols]
    
    # ============================================================
    # STEP 1: IMPUTE NUMERIC COLUMNS (USE SAVED IMPUTER)
    # ============================================================
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    numeric_cols_present = [c for c in numeric_cols if c in df.columns]
    if numeric_cols_present:
        df[numeric_cols_present] = imputer.transform(df[numeric_cols_present])
    
    # ============================================================
    # STEP 2: FEATURE ENGINEERING (creates log features)
    # ============================================================
    engineer = CustomFeatureEngineer()
    df = engineer.transform(df)
    
    # ============================================================
    # STEP 3: SCALING (USE SAVED SCALER)
    # ============================================================
    scale_cols_present = [c for c in scale_cols if c in df.columns]
    if scale_cols_present:
        df[scale_cols_present] = scaler.transform(df[scale_cols_present])
    
    # ============================================================
    # STEP 4: ENCODING, COLUMN ALIGNMENT AND FEATURE SELECTION
    # ============================================================
    encoder = CustomCategoricalEncoder()
    df = encoder.transform(df)
    
    all_columns = selected_features.copy()
    synchronizer = ColumnSynchronizer(ref_columns=all_columns)
    df = synchronizer.transform(df)
    
    for col in ['YearsAtCompanyLog', 'TotalWorkingYearsLog']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median() if len(df) > 1 else 0)
    
    selector = FeatureSelector(features=selected_features)
    df = selector.transform(df)
    
    return df
