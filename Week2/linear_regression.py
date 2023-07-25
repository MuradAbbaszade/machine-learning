import numpy as np
import matplotlib.pyplot as plt
import math

input = np.array([100,200,300,400])
output= np.array([1000,2000,3000,4000])

def predict(input,w,b):
    m = input.shape[0]
    f = np.zeros(m)
    for i in range(m):
        f[i]=w*input[i]+b
    return f

def cost_function(input,output,w,b):
    m=input.shape[0]
    sum=0
    for i in range(m):
        sum+=((input[i]*w+b)-output[i])**2
    return (1/(2*m))*sum

def gradient_function(input,output,w,b):
    dj_b = 0
    dj_w = 0  
    m=input.shape[0]
    for i in range(m):
        dj_w+=(w*input[i]+b-output[i])*input[i]
        dj_b+=(w*input[i]+b-output[i])
    dj_w=dj_w/m
    dj_b=dj_b/m
    print(f"{cost_function(input,output,dj_w,dj_b)}")
    return dj_w,dj_b
        
def gradient_descent(input,output,w,b,alpha,iter_count):
    for i in range(iter_count):
        dj_w,dj_b=gradient_function(input,output,w,b)
        w=w-alpha*dj_w
        b=b-alpha*dj_b
    return w,b


w,b = gradient_descent(input,output,1,2,0.00001,1000)
a = predict(input,w,b)
print(f"{a}")


        

