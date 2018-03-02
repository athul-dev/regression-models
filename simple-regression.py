#Athul Dev - mottu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv",sep='\t')
print(df.head(6))

X=df['PovPct'].values
Y=df['Brth15to17'].values   # can be used with ViolCrime , TeenBrth and so

plt.scatter(X,Y,c='red',label='Scatter Plot')
plt.xlabel('Poverty Range')
plt.ylabel('Birth 15 - 17')
plt.legend()
plt.show()

X=X.reshape(len(X),1)
reg = LinearRegression()
reg=reg.fit(X, Y)
Y_pred = reg.predict(X)

plt.plot(X, Y_pred, color='green', label='Regression Line')
plt.scatter(X, Y, c='red', label='Actual Data')

plt.xlabel('Poverty Rate')
plt.ylabel('Birth 15 - 17')
plt.legend()
plt.show()

print('a is: ',np.round(reg.coef_,2))
print('b is: ',np.round(reg.intercept_,2))
