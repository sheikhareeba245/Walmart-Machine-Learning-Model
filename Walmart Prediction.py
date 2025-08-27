import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
# ---------- Step 1: Load Data ----------
df=pd.read_csv("Data Science with ML/Linear Regression/Mini Project Walmart/data/Walmart_Sales.csv")
print("Dataset Shape: ",df.shape)
print(df.head(10))
# ---------- Step 2: Feature Selection ----------
x=df[["Store","Holiday_Flag","Temperature","Fuel_Price","CPI","Unemployment"]]
y=df["Weekly_Sales"]
# ---------- Step 3: Train-Test Split ----------
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# ---------- Step 4: Train Model ----------
model=LinearRegression()
model.fit(x_train,y_train)
# ---------- Step 5: Predictions ----------
y_pred=model.predict(x_test)
# ---------- Step 6: Evaluation ----------
print("Mean Squared Error (MSE):",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))
# ---------- Step 7: Visualization ----------
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.5,color="blue")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Walmart Sales Prediction-Actual vs Prediction")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"r--")
plt.show()

