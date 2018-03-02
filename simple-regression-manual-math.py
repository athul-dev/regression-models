# Athul Dev - mottu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv",sep='\t')
print(df.head(6))

X=df['PovPct'].values
Y=df['Brth15to17'].values   # can be used with ViolCrime , TeenBrth and so

plt.scatter(X,Y,c='red',label='Scatter Plot')
plt.xlabel('Poverty Range')
plt.ylabel('Birth 15 - 17')
plt.legend()
plt.show()


# mean not required
n=len(X)


sumX=np.sum(X)
sumY=np.sum(Y)
mulXY=np.multiply(X,Y)
sum_mul_XY=np.sum(mulXY)

Xsquare = np.multiply(X,X)
Ysquare = np.multiply(Y,Y)

sum_X_square=np.sum(Xsquare)

a_hat= ((sumX*sumY)-(n*sum_mul_XY))/((sumX*sumX)-(n*sum_X_square))
b_hat= ((sumX*sum_mul_XY)-(sumY*sum_X_square))/((sumX*sumX)-(n*sum_X_square))

print(a_hat)
print(b_hat)



updated_Y= a_hat*X + b_hat
plt.plot(X, updated_Y, color='red', label='Regression Line')
plt.scatter(X, Y, c='blue', label='Actual data')

plt.xlabel('Poverty Rate')
plt.ylabel('15-17 yo Birth rate/100')
plt.legend()
plt.show()

