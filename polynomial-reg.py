#Athul Dev - mottu  

import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv("poly-test.csv",sep='\t')
X=np.array(data['X'].values)
Y=np.array(data['Y'].values)

print(X)
print(Y)


linear = sc.polyfit(X,Y,1)
linear_reg=sc.poly1d(linear)
print(linear)
print(linear_reg)
print("\n")



quadratic=sc.polyfit(X,Y,2)
quadratic_reg=sc.poly1d(quadratic)
print(quadratic)
print(quadratic_reg)
print("\n")

cubic =sc.polyfit(X,Y,3)
cubic_reg=sc.poly1d(cubic)
print(cubic)
print(cubic_reg)
print("\n")

plt.plot(X,Y,'o')
plt.plot(X,linear_reg(X),color='red')
plt.plot(X,quadratic_reg(X),color='green')
plt.plot(X,cubic_reg(X),color='black')
plt.xlabel('Temperature')
plt.ylabel('Temperature Coeffecients')
plt.show()




