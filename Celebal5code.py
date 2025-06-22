import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Combine train and test for uniform preprocessing
train['TrainFlag'] = 1
test['TrainFlag'] = 0
test['SalePrice'] = np.nan  # placeholder target for test data
combined = pd.concat([train, test], axis=0)

# 1️ Handling missing values
# Example: Fill numeric columns with median, categorical with mode
for col in combined.columns:
    if combined[col].dtype == 'object':
        combined[col] = combined[col].fillna(combined[col].mode()[0])
    else:
        combined[col] = combined[col].fillna(combined[col].median())

# 2️ Encode categorical variables
# Label encoding for ordinal-like features or all categorical (simple approach)
label_encoders = {}
for col in combined.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le

# 3️ Feature engineering
# Example: Total square footage
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']

# Example: Age of house at sale
combined['HouseAge'] = combined['YrSold'] - combined['YearBuilt']

# 4️ Feature scaling
numeric_features = combined.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('SalePrice')  # Don't scale target
scaler = StandardScaler()
combined[numeric_features] = scaler.fit_transform(combined[numeric_features])

# 5️ Split back to train and test
processed_train = combined[combined['TrainFlag'] == 1].drop(['TrainFlag'], axis=1)
processed_test = combined[combined['TrainFlag'] == 0].drop(['TrainFlag', 'SalePrice'], axis=1)

# Example output shapes
print(f"Processed train shape: {processed_train.shape}")
print(f"Processed test shape: {processed_test.shape}")

# Save processed files (optional)
processed_train.to_csv('processed_train.csv', index=False)
processed_test.to_csv('processed_test.csv', index=False)
