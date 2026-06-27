Earthquake predictor (Babcsán Gergely, Hiri Olivér)

Earthquake prediction remains one of the most challenging and impactful problems in geoscience, where accurate forecasting can help reduce loss of life and property damage. In this project, machine learning techniques such as Random Forest, Support Vector algorithm, and Linear Regression are applied to analyze seismic data and improve the prediction of earthquake events.

To predict earthquake significance, we used four features: magnitude, latitude, longitude, and depth. The dataset was sourced from historical earthquake records spanning 1995 to 2023, and split into 80% training and 20% testing subsets. Three regression algorithms were trained on the training set, and their performance was compared using the Root Mean Square Error (RMSE) evaluated on the held-out test data.

Data Preparation
The script loads earthquake data from a CSV file and drops several non-numeric or unreliable columns (like title, date_time, alert, location, etc.) to keep only numeric features. Rows with missing values are dropped. The target variable is sig (earthquake significance score), and all remaining columns become the feature set. The data is split into 80% training and 20% testing sets.

Decision Tree Regressor
A single decision tree is trained with max_depth=5 to predict sig. After fitting, it predicts on the test set and reports MSE and R² as baseline performance metrics. The tree structure is also visualized with plot_tree, showing the splits and decision rules learned.

Random Forest Regressor
This section tunes the max_depth hyperparameter by looping through depths 1–20, training a separate Random Forest (100 trees each) at every depth, and recording train/test R² scores. It plots both curves to visually identify where the model starts overfitting (train R² keeps climbing while test R² plateaus or drops), and marks the best-performing depth with a vertical line.The algorithm shows tthat the best depth is 18 but its clear from the previous plot that the best depth should be 5-7 so we retrained the data with max_depth =5.

Linear Regression
This section trains a linear regression model on training data and evaluates its performance on a test set using MAE, MSE, RMSE, and R² metrics. It then displays the model's feature coefficients sorted by absolute magnitude, revealing which features have the strongest influence on the predicted "sig" value, along with the model's intercept. Finally, it generates a scatter plot comparing actual versus predicted values, with points color-coded by absolute residual error, a diagonal "perfect fit" reference line, and a shaded ±RMSE band to visualize prediction accuracy. 

SVM
In this section the code scales the features and target, splits the data into train/test sets, then runs a grid search with 5-fold cross-validation to find the best C, epsilon, and gamma hyperparameters for an SVR model with an RBF kernel, optimizing for R². It then trains the final model using the best parameters found, predicts on the test set, and inverse-transforms the scaled predictions back to the original "sig" scale for evaluation. The model's performance is reported using MAE, RMSE, and R² metrics, printed in a formatted summary block. 