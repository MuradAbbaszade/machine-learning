import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

x_train = np.array([[100,1],[200,3],[300,2],[400,1]])
y_train = np.array([1000,2500,3000,3000])
x_features = np.array(["size","rooms"])

scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train)

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(x_norm, y_train)
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:w: {w_norm}, b:{b_norm}")

y_pred_sgd = sgdr.predict(x_norm)
print(f"{y_pred_sgd}")

fig,ax=plt.subplots(1,2,figsize=(12,4),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(x_features[i])
    ax[i].scatter(x_train[:,i],y_pred_sgd,color="r", label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

x_new = np.array([[400, 1]])
x_new_norm = scaler.transform(x_new)
y_pred_new = sgdr.predict(x_new_norm)
print(f"Tahmin: {y_pred_new}")
