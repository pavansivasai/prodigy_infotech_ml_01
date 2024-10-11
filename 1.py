import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Upload kaggle.json file
from google.colab import files
uploaded = files.upload()

# Step 2: Move kaggle.json to the right directory
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/

# Step 3: Set permissions to the file
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: Download the dataset from Kaggle
!kaggle competitions download -c house-prices-advanced-regression-techniques -p ./data

# Step 5: Unzip the downloaded dataset
import zipfile
with zipfile.ZipFile('./data/house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')

# Load the dataset
df = pd.read_csv('./data/train.csv')

# Select relevant columns
df = df[['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'FullBath', 'BedroomAbvGr', 'SalePrice']]

# Handle missing values by filling them with the median of each column
df = df.fillna(df.median())

# Define features (X) and target (y)
X = df[['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'FullBath', 'BedroomAbvGr']]
y = df['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Display model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
