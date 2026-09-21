from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

# 1. Load the dataset
data = pd.read_csv(BASE_DIR / "house_prices.csv")

# 2. Select input features and target
features = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
X = data[features]
y = data["price"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Test the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("House Price Prediction Model")
print("-" * 32)
print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.3f}")

# 6. Save the trained model for future predictions
model_path = BASE_DIR / "house_price_model.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path.name}")
