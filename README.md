**Walmart Sales Prediction**
This project focuses on predicting Walmart store sales using historical sales data. The aim is to explore data, analyze trends, and build a machine learning model that can forecast sales more accurately.
**Project Overview**
- Performed Exploratory Data Analysis (EDA) to understand sales patterns.
- Visualized key insights such as weekly sales trends, holiday impact, and store performance.
- Built a Linear Regression model to predict sales.
- Evaluated model performance using common metrics like MAE, MSE, and R² Score.
**Dataset**
-The dataset was sourced from Kaggle (Walmart Sales Dataset).
It contains:
- Store → Store number
- Date → Week of sales
- Weekly_Sales → Sales for the given week
- Holiday_Flag → Whether the week was a holiday (1 = Yes, 0 = No)
- Temperature → Average temperature in the region
- Fuel_Price → Cost of fuel in the region
- CPI → Consumer Price Index
- Unemployment → Unemployment rate
**Technologies Used**
- Python
- Pandas & NumPy → Data manipulation & cleaning
- Matplotlib & Seaborn → Data visualization
- Scikit-Learn → Machine Learning model
**Model Building**
- Model Used → Linear Regression
- Input Features → Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
- Target Variable → Weekly_Sales
**Model Performance:**
- MAE (Mean Absolute Error) → Measures average error
- MSE (Mean Squared Error) → Penalizes larger errors
- R² Score → Explains how well the model fits the data
