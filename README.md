✈️ Aircraft Maintenance Predictive Model

This project develops a predictive maintenance model for aircraft engines to estimate their Remaining Useful Life (RUL) using time-series sensor data.
By leveraging machine learning (Random Forest Regressor) and engineered features like rolling statistics, the system forecasts maintenance needs before failures occur — enhancing reliability and reducing downtime.

📊 Project Overview

The project uses three datasets:

PM_train.csv — Training data with engine operational cycles and sensor readings

PM_test.csv — Test data used for performance evaluation

PM_truth.csv — Ground truth RUL values for engines in the test set

The objective is to predict the RUL (Remaining Useful Life) of each engine and visualize the difference between predicted and actual values.

🧩 Workflow
1️⃣ Data Loading

Loaded the provided datasets into Pandas DataFrames:

df_train = pd.read_csv('/content/PM_train.csv')
df_test = pd.read_csv('/content/PM_test.csv')
df_truth = pd.read_csv('/content/PM_truth.csv')

2️⃣ Data Preprocessing

Verified that there were no missing values.

Identified and removed constant and low-variance features such as:

setting3, s1, s5, s10, s16, s18, s19


These features add minimal information for model learning.

3️⃣ Feature Engineering

RUL Calculation:
For each engine id, the RUL was calculated as:

𝑅
𝑈
𝐿
=
max(cycle)
−
current cycle
RUL=max(cycle)−current cycle

Rolling Statistics:
Generated rolling mean and rolling standard deviation over a window of 10 cycles for all sensor readings:

df_train[f'{feature}_rolling_mean_10']
df_train[f'{feature}_rolling_std_10']


These features capture short-term trends in engine performance.

4️⃣ Model Training

Algorithm Used: RandomForestRegressor

Split: 80% training / 20% validation

Target Variable: Remaining Useful Life (RUL)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

5️⃣ Prediction

Predicted RUL on the test dataset.

Filled any missing rolling-window values using mean imputation.

Stored results in a new column:

df_test['predicted_RUL'] = model.predict(X_test)

6️⃣ Evaluation

Compared predicted and true RUL values for the last cycle of each engine.

Metric	Value
Root Mean Squared Error (RMSE)	34.07
Mean Absolute Error (MAE)	23.91

These metrics indicate a strong baseline performance for maintenance prediction.

7️⃣ Visualization

Scatter Plot: True vs Predicted RUL


Line Plots: RUL trend over cycles for selected engines


📈 Results Summary

No missing data found in any dataset.

Removed constant and low-variance features to prevent overfitting.

Engineered 30+ rolling features that capture performance trends.

Achieved RMSE = 34.07 and MAE = 23.91 on test data.

Visualized RUL degradation and prediction accuracy over time.

🚀 Future Improvements

🔧 Hyperparameter Tuning: Optimize number of trees, depth, and sampling parameters in the Random Forest.

🤖 Model Comparison: Explore LSTM, XGBoost, or Temporal Convolutional Networks for time-series modeling.

🧠 Feature Selection: Evaluate correlation and SHAP feature importance.

📅 Predictive Scheduling: Integrate with maintenance planning dashboards for real-time insights.

🧰 Tech Stack
Category	Tools / Libraries
Programming	Python 3
Data Handling	Pandas, NumPy
Modeling	scikit-learn
Visualization	Matplotlib
Notebook Environment	Jupyter / Google Colab
 
