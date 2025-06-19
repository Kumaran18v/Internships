import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#https://drive.google.com/file/d/1aaQNv8wCIsYr7LTGlmnG86dlZsEQN9vM/view?usp=drive_link
data = pd.read_csv("/content/drive/MyDrive/Data/salary_Data.csv")
print(data.head())
print(data.describe())
print(data.info())

sns.scatterplot(x='YearsExperience', y='Salary', data=data)
plt.title("Years of Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

X = data[['YearsExperience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Prediction Line')
plt.title("Linear Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
compare.plot(kind='bar', figsize=(10, 6))
plt.title("Actual vs Predicted Salaries")
plt.xlabel("Sample")
plt.ylabel("Salary")
plt.show()

def predict_salary():
    try:
        experience = float(input("Enter years of experience: "))
        input_df = pd.DataFrame({'YearsExperience': [experience]})
        salary = model.predict(input_df)
        print(f"Predicted Salary: {salary[0]:.2f}")
    except ValueError:
        print("Please enter a valid number.")

predict_salary()
