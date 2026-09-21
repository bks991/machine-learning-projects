# House Price Prediction

A beginner-friendly machine learning project that predicts a house price using **Python** and **scikit-learn**.

## Project Goal

The project trains a Linear Regression model using four features:

- House area in square feet
- Number of bedrooms
- Number of bathrooms
- Age of the house

The dataset in this repository is **synthetic** and is included only for learning and demonstration.

## Technologies

- Python
- pandas
- scikit-learn
- joblib

## Project Files

```text
house-price-prediction/
├── house_prices.csv
├── train_model.py
├── predict_house.py
├── requirements.txt
└── README.md
```

## How It Works

1. `train_model.py` loads the CSV dataset.
2. The data is split into training and testing sets.
3. A Linear Regression model is trained.
4. The model is evaluated using Mean Absolute Error and R² score.
5. The trained model is saved as `house_price_model.joblib`.
6. `predict_house.py` uses the saved model to estimate a price for a new house.

## Installation

Open a terminal inside the project folder and run:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

## Make a Prediction

After training the model, run:

```bash
python predict_house.py
```

Then enter the requested house details.

## Example

```text
Area in square feet: 1600
Number of bedrooms: 3
Number of bathrooms: 2
Age of house in years: 10
```

The program will print an estimated house price.

## What I Learned

- How to load and prepare a dataset with pandas
- How to split data into training and testing sets
- How to train a Linear Regression model
- How to evaluate a regression model
- How to save and reuse a trained machine learning model

## Disclaimer

This project uses synthetic data and is for educational purposes only. It should not be used for real property valuation.
