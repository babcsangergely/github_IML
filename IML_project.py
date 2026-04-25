import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("earthquake_data.csv", on_bad_lines="skip")

# Drop non-numeric / problematic columns
df = df.drop(columns=[
    'title',  'date_time', 'cdi', 'mmi', 'alert', 'tsunami',
        'net', 'nst', 'dmin', 'gap', 'magType',  
        'location', 'continent', 'country'
])

# Remove missing values
df = df.dropna()

# Target
y = df["sig"]

# Features
X = df.drop(columns=["sig"])
feature_names = X.columns.astype(str)
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Simple decision tree
reg = DecisionTreeRegressor(max_depth=5, random_state=42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test) # Predict 

mse = mean_squared_error(y_test, y_pred) # Evaluate
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.3f}")
print(f"R² score: {r2:.3f}")

plt.figure(figsize=(14, 8)) # Plot tree
plot_tree(reg, feature_names=feature_names, filled=True)
plt.show()




#Random Forest
# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=100,     # number of trees
    max_depth=5,       # limit tree depth to prevent overfitting
    random_state=42,
                 
)

rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"R² score: {r2:.3f}")

#plotting the feature importance

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
plt.show()