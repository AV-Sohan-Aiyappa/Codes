
import numpy as np
import pandas as pd
from  sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

df_x = pd.DataFrame(california.data,columns=california.feature_names)
df_y = pd.DataFrame(california.target)

reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state = 83)

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)


from sklearn.metrics import mean_squared_error
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))