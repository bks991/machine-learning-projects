from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"

if not MODEL_PATH.exists():
    print("Model not found.")
    print("Please run: python train_model.py")
    raise SystemExit

model = joblib.load(MODEL_PATH)

print("Enter the house details:")
area_sqft = float(input("Area in square feet: "))
bedrooms = int(input("Number of bedrooms: "))
bathrooms = int(input("Number of bathrooms: "))
age_years = int(input("Age of house in years: "))

house = pd.DataFrame(
    [[area_sqft, bedrooms, bathrooms, age_years]],
    columns=["area_sqft", "bedrooms", "bathrooms", "age_years"],
)

predicted_price = model.predict(house)[0]

print(f"\nEstimated house price: ${predicted_price:,.2f}")
print("\nNote: This is a learning project using synthetic data, not a real property valuation.")
