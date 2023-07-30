import numpy as np
import matplotlib.pyplot as plt
import math

input = np.array([10,20,30,40])
output= np.array([0,0,1,1])

def func_z(input,w,b):
    return np.dot(input,w)+b

def sigmoid_function(z):
    return 1/(1+np.exp(-z))
    
def cost_function(input,output, w, b):
    m = input.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(input[i],w) + b
        f_wb_i = sigmoid_function(z_i)
        cost +=  -output[i]*np.log(f_wb_i) - (1-output[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

def gradient_function(input,output,w,b):
    dj_w=0
    dj_b=0
    m=input.shape[0]
    for i in range(m):
        z=func_z(input[i],w,b)
        dj_w+=(sigmoid_function(z)-output[i])*input[i]
        dj_b+=sigmoid_function(z)-output[i]
    dj_w=dj_w/m
    dj_b=dj_b/m
    return dj_w,dj_b

def gradient_descent(input,output,w,b,alpha,iter_count):
    for i in range(iter_count):
        dj_w,dj_b=gradient_function(input,output,w,b)
        w=w-alpha*dj_w
        b=b-alpha*dj_b
        cost = cost_function(input, output, w, b)
        print(f"{cost}")
    return w,b

w,b=gradient_descent(input,output,5,5,0.01,5000)
print(f"{w,b}")
z=func_z(input,w,b)
print(f"{sigmoid_function(z)}")
    

        

