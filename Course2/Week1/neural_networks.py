import numpy as np
X_tst = np.array([
    [200,13.9],
    [200,17],
[300,16]])
def g(z):
    return 1/(1+np.exp(-z))

def my_dense(x,W,b):
    units = W.shape[1]
    y=np.zeros(units)
    for i in range(units):
        w=W[:,i]
        z=np.dot(w,x)+b[i]
        y[i]=g(z)
    return y

def my_sequential(x,W1,b1,W2,b2):
    a=my_dense(x,W1,b1)
    a2=my_dense(a,W2,b2)
    return a2

W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], 
                    [-27.59], 
                    [-32.56]] )
b2_tmp = np.array( [15.41] )

def my_predict(X,W1,b1,W2,b2):
    m=X.shape[0]
    result=np.zeros((m,1))
    for i in range(m):
        result[i,0]=my_sequential(X[i],W1,b1,W2,b2)
    return result
    
a = my_predict(X_tst,W1_tmp,b1_tmp,W2_tmp,b2_tmp)
print(f"{a}")
        
        
        

