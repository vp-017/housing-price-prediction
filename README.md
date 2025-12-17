Housing Price Prediction using Regression Models

This project aims to predict house prices using multiple regression techniques and compare their performance on the same dataset. The objective is to analyze how different regression models behave when applied to real housing data and to identify the most suitable model for accurate price prediction.

Project Structure

housing-price-prediction/
data/Housing.csv
src/preprocessing.py
src/simple_linear_regression.py
src/multiple_linear_regression.py
src/polynomial_regression.py
results/comparison.csv
README.md
requirements.txt
.gitignore

Dataset

The dataset used in this project is Housing.csv. It contains various features related to houses such as area, number of rooms, and location-based attributes. The target variable is the house price.

Regression Models Implemented

The following regression models are implemented and evaluated:

Simple Linear Regression

Multiple Linear Regression

Polynomial Regression

All models use the same preprocessing pipeline to ensure a fair and consistent comparison.

Data Preprocessing

The preprocessing pipeline includes handling missing values, encoding categorical variables, splitting the data into training and testing sets, and applying feature transformations where required. These steps ensure the data is suitable for regression modeling.

Model Evaluation

Model performance is evaluated using standard regression metrics:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

A comparative summary of all models is stored in the results/comparison.csv file.

Execution Instructions

Install the required dependencies using pip install -r requirements.txt.
Run any regression model from the src directory, for example:
python src/simple_linear_regression.py
python src/multiple_linear_regression.py
python src/polynomial_regression.py

Project Outcome

This project demonstrates the application of multiple regression techniques on a single dataset, highlights the strengths and limitations of each model, and provides a structured comparison to support model selection.
