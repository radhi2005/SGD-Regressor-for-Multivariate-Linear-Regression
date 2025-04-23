# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the necessary libraries
Load and analyse the dataset
Convert the dataset into pandas dataframe for easier access
Go with preprocessing if required
Assign the input features and target variable
Standardize the input features using StandardScaler
Train the model using SGDRegressor and MultiOutputRegressor
Now test the model with new values
And measure the accuracy using MSE

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: RADHIMEENA M
RegisterNumber: 212223040159 
*/
import numpy as np
        import pandas as pd
        from sklearn.datasets import fetch_california_housing
        from sklearn.linear_model import SGDRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import StandardScaler
        
        data=fetch_california_housing()
        print(data)
        
        df=pd.DataFrame(data.data,columns=data.feature_names)
        df['target']=data.target
        print(df.head())
        print(df.tail())
        print(df.info())
        
        x=df.drop(columns=['AveOccup','target'])
        y=df['target']
        
        print(x.shape)
        print(y.shape)
        print(x.info())
        print(y.info())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        
        scaler_x=StandardScaler()
        x_train=scaler_x.fit_transform(x_train)
        x_test=scaler_x.transform(x_test)
        scaler_y=StandardScaler()
        
        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        y_train=scaler_y.fit_transform(y_train)
        y_test=scaler_y.transform(y_test)
        
        sgd=SGDRegressor(max_iter=1000,tol=1e-3)
        multi_output_sgd=MultiOutputRegressor(sgd)
        multi_output_sgd.fit(x_train,y_train)
        y_pred=multi_output_sgd.predict(x_test)
        y_pred=scaler_y.inverse_transform(y_pred)
        y_test=scaler_y.inverse_transform(y_test)
        
        mse=mean_squared_error(y_test,y_pred)
        print("Mean Squared Error:",mse)

        print("\nPredictions:\n",y_pred[:5])
```

## Output:
![image](https://github.com/user-attachments/assets/c4d5e0f6-6155-48a9-a6a5-09b8f61d452e)
![image](https://github.com/user-attachments/assets/33fdbe23-c9f3-4965-b999-a88698e48baa)
![image](https://github.com/user-attachments/assets/75f6c9d8-479b-4ba1-a93b-c4ce0de28cba)
![image](https://github.com/user-attachments/assets/68173769-2fe1-48e9-b598-832ee2d0e8f1)
![image](https://github.com/user-attachments/assets/b0e70f52-73a6-487b-aab2-a75b33afb57b)
![image](https://github.com/user-attachments/assets/edccde84-a249-4b43-8e11-1c7adbf6e43f)
![image](https://github.com/user-attachments/assets/10c05b51-8ab1-4005-bdf4-553a5c10b721)
![image](https://github.com/user-attachments/assets/2f88500f-03e1-4c85-9000-aed2c0b60550)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
