#Athul Dev - mottu
# We will have to use the R suqare computation to obtain the X and Y values in the normal from
# thats the reason we are not getting the right coeffecients. Please feel free to improve on it.
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

df = pd.read_csv("multivar.csv",sep='\t')
print(df.head(6))

X=df['Mech Apt'].values
X1=df['Consc'].values
Y=df['Job Perf'].values

# plt.scatter(X,Y,c='red',label='Scatter Plot')
# plt.xlabel('')
# plt.ylabel('')
# plt.legend()
# plt.show()
#


n=len(X)

X_mean=np.mean(X)
X1_mean=np.mean(X1)
Y_mean=np.mean(Y)

sumX=np.sum(X)
sumX1=np.sum(X1)
sumY=np.sum(Y)
#
mulXY=np.multiply(X,Y)
mulX1Y=np.multiply(X1,Y)
#
sum_mul_XY= np.sum(mulXY) # - ((sumX*sumY)/n)
sum_mul_X1Y=np.sum(mulX1Y) #- ((sumX1*sumY)/n)
#
Xsquare = np.multiply(X,X)
X1square = np.multiply(X1,X1)
Ysquare = np.multiply(Y,Y)
#
sum_X_square=np.sum(Xsquare)
sum_X1_square=np.sum(X1square)

mul_XX1 = np.multiply(X,X1)
sum_mul_XX1=np.sum(mul_XX1) #- ((sumX*sumX1)/n)


b1 = (((sum_X1_square)*(sum_mul_XY))-((sum_mul_XX1)*(sum_mul_X1Y))) / (((sum_X_square)*(sum_X1_square))-(sum_mul_XX1*sum_mul_XX1))
b2 = (((sum_X_square)*(sum_mul_X1Y))-((sum_mul_XX1)*(sum_mul_XY))) / (((sum_X_square)*(sum_X1_square)) - (sum_mul_XX1*sum_mul_XX1))
a=Y_mean-(b1*X_mean)-(b2*X1_mean)

# print("sum of Y =",Y)
# print("sum of X =",X)
# print("x mean=",X_mean)
# print("x1 mean=",X1_mean)
# print("y mean=",Y_mean)
# print("sum of X square =",sum_X_square)
# print("sum of X1 square =",sum_X1_square)
# print("sum of mul of X1*Y =",sum_mul_X1Y)
# print("sum of mul of X*Y  =",sum_mul_XY)
# print("sum of mul of X*X1  =",sum_mul_XX1)
# print("sum of mul of X*X1 square  =",sum_mul_XX1*sum_mul_XX1)


print("Coeffecient of X1",b1)
print("Coeffecient of X2 ",b2)
print("Constant Part",a)

updated_y = b1*X + b2*X1 + a

