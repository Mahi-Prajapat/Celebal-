
### 1. Importing Libraries and Loading Data


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic_data = pd.read_csv(url)


### 2. Understanding Data Distributions


# Plotting histograms for numerical features
titanic_data.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()


### 3. Identifying Missing Values


# Checking for missing values
missing_values = titanic_data.isnull().sum()
missing_values[missing_values > 0]


### 4. Detecting Outliers


# Box plot for 'Fare' and 'Age'
plt.figure(figsize=(12, 6))
sns.boxplot(data=titanic_data[['Fare', 'Age']])
plt.title('Box Plot of Fare and Age')
plt.show()


### 5. Uncovering Relationships Between Variables


# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = titanic_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


### 6. Additional Visualizations


# Count plot for survivors by gender
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survivors Count by Gender')
plt.show()


### Conclusion


# Correlation heatmap
# plt.figure(figsize=(10, 8))
# correlation_matrix = titanic_data.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()
