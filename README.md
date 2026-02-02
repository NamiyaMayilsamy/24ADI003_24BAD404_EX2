# 24ADI003_24BAD404_EX2
MACHINE LEARNING EX2
SCENARIO 1: Ocean Water Temperature Prediction

Dataset:
Kaggle – Public Dataset
https://www.kaggle.com/datasets/sohier/calcofi

This scenario involves predicting ocean water temperature using environmental and depth-related factors. The dataset is loaded into Pandas and explored using basic exploratory functions to understand its structure, data types, and missing values.

Target Variable

Water Temperature (T_degC)

Sample Input Features

Depth (m)

Salinity

Oxygen

Latitude

Longitude

Description

Missing values in the dataset are handled using mean or median imputation to ensure data completeness. Feature scaling is applied using StandardScaler to normalize numerical inputs. The dataset is then split into training and testing sets.

A Linear Regression model is trained using Scikit-learn to predict water temperature. Model performance is evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score. Visualizations such as Actual vs Predicted values and residual plots are used to analyze prediction accuracy and error distribution.

To improve performance, feature selection techniques and regularization methods (Ridge and Lasso) are applied and compared.

SCENARIO 2: LIC Stock Price Movement Classification

Dataset:
Kaggle – Public Dataset
https://www.kaggle.com/datasets/debashis74017/lic-stock-price-data

This scenario focuses on predicting whether the LIC stock price will increase or decrease based on historical trading data. The dataset is explored to understand stock price behavior and identify missing values.

Target Variable (Derived)

Price Movement

1 → Closing Price > Opening Price

0 → Closing Price ≤ Opening Price

Input Features

Open

High

Low

Volume

Description

After loading the dataset into Pandas, a binary target variable is created based on opening and closing prices. Missing values are handled, and numerical features are scaled for better model convergence.

A Logistic Regression model is trained to classify stock price movement. Model performance is evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix. Visualization techniques such as the ROC curve and feature importance analysis are used to understand model behavior.

To enhance classification performance, hyperparameter tuning (regularization strength C and penalty type) and regularization techniques are applied.

OUTCOME

This lab exercise demonstrates practical implementation of regression techniques on real-world datasets. Linear Regression is used for continuous value prediction, while Logistic Regression is applied to binary classification problems. The exercise emphasizes data preprocessing, model evaluation, visualization, and optimization, providing a strong foundation for real-world machine learning applications.
