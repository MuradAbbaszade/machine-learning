import numpy as np
import matplotlib.pyplot as plt

input_array = np.array([100,200,300])
output_array = np.array([10000,20000,40000])

w=200
b=300

def cost_function(input_arr,output_arr,w,b):
    length=len(input_arr)
    sum=0
    for i in range(length):
        f_wb=input_arr[i]*w+b
        diff =(f_wb-output_arr[i])**2
        sum+=diff
    j_wb=(1/2*length)*sum
    return j_wb

a = cost_function(input_array,output_array,100,200)

def computing_f(input_arr,w,b):
    m=len(input_arr)
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*input_arr[i]+b
    return f_wb


output_f = computing_f(input_array,w,b)
plt.scatter(input_array,output_array,marker="x",c="r")
plt.title("House price prediction")
plt.xlabel("House size")
plt.ylabel("House price")
plt.plot(input_array,output_f,c="y")
plt.show()


