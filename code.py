# Step 1: Install libraries (needed in Colab only)
!pip install scikit-learn pandas

# Step 2: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 3: Load dataset
# Upload your CSV file in Colab (left side panel → Files → Upload)
from google.colab import files
uploaded = files.upload()

# Replace 'your_file.csv' with your uploaded file name
df = pd.read_csv("Housing.csv")




# Step 4: Features (X) and Target (y)
# Example: Suppose "Price" is the target column
X = df.drop("price", axis=1)   # all features except target
y = df["price"]                # target column

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("\n✅ Model Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Step 9: Test with custom input
# Make sure the order of features matches your CSV columns (except target)
sample_house = [X_test.iloc[0].values]  # taking first test sample
predicted_price = model.predict(sample_house)
print("\n🏠 Predicted Price for sample house:", predicted_price[0])
