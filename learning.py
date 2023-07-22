import numpy as np
import matplotlib.pyplot as plt

input_array = np.array([100,200,300])
output_array = np.array([10000,20000,40000])

w=100
b=200

plt.scatter(input_array,output_array,marker="x",c="r")
plt.title("House price prediction")
plt.xlabel("House size")
plt.ylabel("House price")
plt.plot(input_array,output_array)
plt.show()


y_hat = computing_f(input_array,w,b)
plt.scatter(input_array,y_hat,marker="x",c="r")
plt.title("House price prediction")
plt.xlabel("House size")
plt.ylabel("House price")
plt.plot(input_array,y_hat)
plt.show()

def computing_f(x,w,b):
    m=len(x)
    f=np.zeros(m)
    for i in range(m):
        f[i]=w*x[i]+b
    return f
    

