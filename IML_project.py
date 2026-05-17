import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler




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


# ── 2. Define features and target ────────────────────────────────────────────
 
TARGET   = "sig"
FEATURES = [col for col in df.columns if col != TARGET]
 
X = df[FEATURES]
y = df[TARGET]
 
print(f"Dataset shape : {df.shape}")
print(f"Features used : {FEATURES}\n")
 
# ── 3. Scale features ─────────────────────────────────────────────────────────
# Linear regression isn't sensitive to scale for coefficient estimation,
# but scaling makes coefficients comparable and improves numerical stability.
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# ── 4. Train / test split ─────────────────────────────────────────────────────
 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
 
# ── 5. Fit model ──────────────────────────────────────────────────────────────
 
model = LinearRegression()
model.fit(X_train, y_train)
 
# ── 6. Evaluate ───────────────────────────────────────────────────────────────
 
y_pred = model.predict(X_test)
 
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
 
print("=" * 50)
print("  Linear Regression Results (sig prediction)")
print("=" * 50)
print(f"  MAE  : {mae:.2f}")
print(f"  MSE  : {mse:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print("=" * 50)
 
# 5-fold cross-validation
cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
print(f"\n  5-fold CV R² : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}\n")
 
# ── 7. Feature coefficients ───────────────────────────────────────────────────
# Scaled coefficients show the relative importance of each feature.
 
coef_df = pd.DataFrame({
    "feature"    : FEATURES,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)
 
print("Feature coefficients (sorted by absolute value):")
print(coef_df.to_string(index=False))
print(f"\nIntercept: {model.intercept_:.2f}")
 

fig, ax = plt.subplots(figsize=(8, 7))
 
# --- colour-code by residual magnitude ---
residuals     = y_test - y_pred
abs_residuals = np.abs(residuals)
norm          = plt.Normalize(abs_residuals.min(), abs_residuals.max())
colours       = plt.cm.RdYlGn_r(norm(abs_residuals))  # green=small error, red=large
 
scatter = ax.scatter(
    y_test, y_pred,
    c=abs_residuals, cmap="RdYlGn_r",
    alpha=0.65, edgecolors="none", s=40, zorder=3
)
 
# --- perfect-fit diagonal ---
lims = [min(y_test.min(), y_pred.min()) - 20,
        max(y_test.max(), y_pred.max()) + 20]
ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect fit", zorder=2)
 
# --- ±RMSE band around the diagonal ---
ax.fill_between(lims,
                [l - rmse for l in lims],
                [l + rmse for l in lims],
                alpha=0.08, color="steelblue", label=f"±RMSE band ({rmse:.0f})")
 
# --- colour bar ---
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Absolute error", fontsize=11)
 
# --- metric annotation box ---
stats_text = (
    f"R²   = {r2:.3f}\n"
    f"RMSE = {rmse:.1f}\n"
    f"MAE  = {mae:.1f}"
)
ax.text(
    0.04, 0.96, stats_text,
    transform=ax.transAxes,
    fontsize=10, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
              edgecolor="#cccccc", alpha=0.85)
)
 
# --- labels ---
ax.set_xlabel("Actual sig", fontsize=12)
ax.set_ylabel("Predicted sig", fontsize=12)
ax.set_title("Linear Regression — Actual vs Predicted (sig)", fontsize=13, pad=12)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
 
plt.tight_layout()
plt.show()

# ── 1. Load & prepare ────────────────────────────────────────────────────────

df = pd.read_csv("earthquake_data.csv")

FEATURES = ["magnitude", "longitude", "latitude", "depth"]
TARGET   = "sig"

df_clean = df[FEATURES + [TARGET]].dropna()
print(f"Samples after dropping NaNs: {len(df_clean)}  (original: {len(df)})")
print(f"\nTarget (sig) stats:\n{df_clean[TARGET].describe().round(2)}\n")

X = df_clean[FEATURES].values
y = df_clean[TARGET].values

# ── 2. Scale ─────────────────────────────────────────────────────────────────

scaler_X = StandardScaler()
scaler_y = StandardScaler()          # scale target too — helps SVR greatly

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ── 3. Train / test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ── 4. Grid search for best C & epsilon ──────────────────────────────────────

print("Running GridSearchCV (RBF kernel) …")
param_grid = {
    "C":       [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.5],
    "gamma":   ["scale", "auto"],
}
grid = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_train, y_train)
best_params = grid.best_params_
print(f"Best params: {best_params}")
print(f"Best CV R²:  {grid.best_score_:.4f}\n")

# ── 5. Final model ────────────────────────────────────────────────────────────

model = grid.best_estimator_

y_pred_scaled = model.predict(X_test)

# Inverse-transform back to original sig scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print("=" * 50)
print("  SVR Regression Results (sig prediction)")
print("=" * 50)
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print("=" * 50)

# 5-fold CV on full data
cv_r2 = cross_val_score(model, X_scaled, y_scaled, cv=5, scoring="r2")
print(f"\n  5-fold CV R² : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")