# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the encoded car dataset and preprocess it by separating features (X) and target variable (price).
2. Apply StandardScaler to normalize both input features and target values.
3. Split the dataset into training and testing sets using train_test_split.
4. Create Polynomial Regression pipelines with Ridge regression, Lasso regression, and Elastic Net models and train them on the training data.
5. Evaluate each model using MSE, MAE, and R² score, then compare results using bar charts. 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
df=pd.read_csv('encoded_car_data.csv')
df.head()
df=pd.get_dummies(df,drop_first=True)
X=df.drop('price',axis=1)
y=df['price']
scaler=StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(y.values.reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
models={
    "Ridge":Ridge(alpha=1.0),
    "Lasso":Lasso(alpha=1.0),
    "ElasticNet":ElasticNet(alpha=1.0,l1_ratio=0.5)
}
results={}
for name,model in models.items():
    pipeline=Pipeline([
        ('poly',PolynomialFeatures(degree=2)),
    ('regressor',model)
    ])
    pipeline.fit(X_train,y_train)
    predictions=pipeline.predict(X_test)
    mse=mean_squared_error(y_test,predictions)
mae=mean_absolute_error(y_test,predictions)
r2=r2_score(y_test,predictions)
results[name]={'MSE':mse,'MAE':mae,'R2 Score':r2}
print('Name: K SADHANA')
print('Reg. No:212225240128')
for model_name,metrics in results.items():
    print(f"{model_name} -Mean Squared Error: {metrics['MSE']:.2f},R2 Score: {metrics['R2 Score']:.2f},Mean Absolute Error: {metrics['MAE']:.2f}")
results_df=pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'},inplace=True)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.barplot(x='Model',y='MSE',data=results_df,palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.subplot(1,2,2)
sns.barplot(x='Model',y='R2 Score',data=results_df,palette='viridis')
plt.title('R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```

## Output:
<img width="919" height="83" alt="image" src="https://github.com/user-attachments/assets/a9afe2f8-abfe-4f5e-a35c-17a730262e71" />


<img width="786" height="640" alt="image" src="https://github.com/user-attachments/assets/84778f53-e29d-4451-803b-a03a088274be" />


<img width="559" height="595" alt="image" src="https://github.com/user-attachments/assets/36bf01e1-6857-48e2-92c1-f6efbbc0ea8c" />

## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
