import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.optimize import minimize
import pandas as pd
import matplotlib

matplotlib.use('Agg')

def gm11(x0):
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.vstack([-z1, np.ones(n-1)]).T
    Y = x0[1:]
    A = inv(B.T @ B) @ B.T @ Y
    a, b = A[0], A[1]
    return a, b

def gm11_predict(x0, a, b, n):
    x1_hat = (x0[0] - b/a) * np.exp(-a * np.arange(n)) + b/a
    x0_hat = np.diff(np.concatenate(([x0[0]], x1_hat)))
    return x0_hat

def dgm(x0):
    n = len(x0)
    x1 = np.cumsum(x0)
    B = np.vstack([x1[:-1], np.ones(n-1)]).T
    Y = x1[1:]
    A = inv(B.T @ B) @ B.T @ Y
    b1, b2 = A[0], A[1]
    return b1, b2

def dgm_predict(x0, b1, b2, n):
    x1_hat = [x0[0]]
    for k in range(1, n):
        x1_hat.append(b1 * x1_hat[-1] + b2)
    x0_hat = np.diff(x1_hat, prepend=x1_hat[0])
    return x0_hat

def osdgm(x0, b3):
    n = len(x0)
    x1 = np.cumsum(x0)
    B = np.vstack([x1[:-1], np.ones(n-1)]).T
    Y = x1[1:]
    A = inv(B.T @ B) @ B.T @ Y
    b1, b2 = A[0], A[1]
    return b1, b2, b3

def osdgm_predict(x0, b1, b2, b3, n):
    x1_hat = [x0[0]]
    for k in range(1, n):
        x1_hat.append(b1 * x1_hat[-1] + b2 + b3)
    x0_hat = np.diff(x1_hat, prepend=x1_hat[0])
    return x0_hat

def mse_loss(b3, x0, true_values):
    b1, b2, _ = osdgm(x0, b3)
    osdgm_pred = osdgm_predict(x0, b1, b2, b3, len(true_values))  # تغییر این خط
    mse = np.mean((osdgm_pred - true_values) ** 2)
    return mse

true_values = np.array([23.4805, 35.6746, 54.2014, 82.3499, 125.1167])

x0 = np.array([21.1, 26.6, 36.1, 52.3, 80.1, 126.8])

result = minimize(mse_loss, x0=0.0, args=(x0, true_values))
optimal_b3 = result.x[0]

b1, b2, b3 = osdgm(x0, optimal_b3)
osdgm_prediction = osdgm_predict(x0, b1, b2, b3, len(true_values))  # تغییر این خط

print("Optimal b3: ", optimal_b3)
print("OSDGM Prediction: ", osdgm_prediction)

df = pd.DataFrame({
    'Original Data': x0[:5],  # فقط 5 مقدار اول
    'True Values': true_values,
    'OSDGM Prediction': osdgm_prediction
})

df.to_csv('osdgm_prediction_comparison.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(x0[:5], label='Original Data', marker='o')
plt.plot(true_values, label='True Values', marker='s')
plt.plot(osdgm_prediction, label='OSDGM Prediction', marker='d')
plt.legend()
plt.title('OSDGM Prediction vs True Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('osdgm_prediction_comparison.png')
