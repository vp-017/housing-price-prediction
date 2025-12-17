# ================= IMPORT LIBRARIES =================
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ================= LOAD DATA =================
dataset = pd.read_csv("Housing.csv")

X = dataset.iloc[:, 1:].values   # Features
y = dataset.iloc[:, 0].values   # Target (price)


# ================= ONE-HOT ENCODING =================
ct = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(drop="first"), [4, 5, 6, 7, 8, 10, 11])
    ],
    remainder="passthrough"
)

X = ct.fit_transform(X)
X = np.array(X)


# ================= TRAIN-TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# ================= TRAIN MODEL =================
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ================= MODEL EVALUATION =================
y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)
print("\n--- Predicted vs Actual Prices (Test Set) ---")
print(
    np.concatenate(
        (y_pred.reshape(-1, 1),
         y_test.reshape(-1, 1)),
        axis=1
    )
)


# ================= SAFE INPUT FUNCTION =================
def yes_no_input(prompt):
    while True:
        value = input(prompt).strip().lower()
        if value in ["yes", "no"]:
            return value
        print("❌ Please enter 'yes' or 'no'")


# ================= USER INPUT =================
print("\n--- Enter New House Details ---")

area = int(input("Area (sq ft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
stories = int(input("Stories: "))

mainroad = yes_no_input("Main road (yes/no): ")
guestroom = yes_no_input("Guest room (yes/no): ")
basement = yes_no_input("Basement (yes/no): ")
hotwaterheating = yes_no_input("Hot water heating (yes/no): ")
airconditioning = yes_no_input("Air conditioning (yes/no): ")

parking = int(input("Parking spaces: "))
prefarea = yes_no_input("Preferred area (yes/no): ")

while True:
    furnishingstatus = input(
        "Furnishing (furnished / semi-furnished / unfurnished): "
    ).lower()
    if furnishingstatus in ["furnished", "semi-furnished", "unfurnished"]:
        break
    print("❌ Invalid furnishing type")


# ================= PREDICTION =================
new_house = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}])

new_house_encoded = ct.transform(new_house)
predicted_price = regressor.predict(new_house_encoded)

print("\n🏠 Predicted House Price: ₹", round(predicted_price[0], 2))
