import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def f_wb_x(x, w_vec, b_val):
    return np.dot(x, w_vec) + b_val

def compute_cost(x_values: np.ndarray, y_values: np.ndarray, w_vec: np.ndarray, b_val: int, reg_param: float) -> float:
    cost = 0.0
    m, n = x_values.shape
    
    for i in range(m):
        cost += (f_wb_x(x_values[i], w_vec, b_val) - y_values[i])**2
    cost /= 2*m
    
    reg_cost = 0
    for i in range(n):
        reg_cost += w_vec[i]**2
    reg_cost *= reg_param/(2*m)
    
    return cost + reg_cost

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n, ))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                 
    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_db, dj_dw

def gradient_descent(x_values: np.ndarray, y_values: np.ndarray, w_vec: np.ndarray, b_val: int, alpha: float, iterations: int, reg_param: float):
    
    temp_w = w_vec.copy()
    temp_b = b_val
    
    J_history = []
    
    for _ in tqdm(range(iterations)):
        dj_db,dj_dw = compute_gradient(x_values, y_values, temp_w, temp_b)
        
        temp_w = temp_w - alpha * dj_dw
        temp_b = temp_b - alpha * dj_db
        
        J_history.append(compute_cost(x_values, y_values, temp_w, temp_b, reg_param))
        
    return J_history, temp_w, temp_b


doc = pd.read_csv("housing - Copy.csv")
doc = doc.dropna()
doc = doc.drop_duplicates()

y_values = doc[doc.columns[-1]].values

doc = doc.drop("median_house_value", axis = 1)
x_values = doc.values

scaler = StandardScaler()
x_values = scaler.fit_transform(x_values)

w_vec = np.zeros(x_values.shape[1])
b_val = 0

# w_vec = np.array([-85610.16499828, -90812.72839684,  14579.63057403, -18024.80275216, 47949.19371492, -43500.08748576,  18247.29674529,  76534.14588457])
# b_val = 206864.41315519018
# index = 100
# print(f"Predicted value: {f_wb_x(x_values[index], w_vec, b_val)}")
# print(f"Actual value: {y_values[index]}")
# print(f"Difference = {f_wb_x(x_values[index], w_vec, b_val) - y_values[index]}")

reg_param = 0
alpha = 0.5
iterations = 100

J_hist, w_vec, b_val = gradient_descent(x_values, y_values, w_vec, b_val, alpha, iterations, reg_param)

print(f"W: {w_vec}")
print(f"B: {b_val}")

index = 100
print(f"Difference = {f_wb_x(x_values[index], w_vec, b_val) - y_values[index]}")
print(f"Final Cost: {J_hist[-1]}")

plt.plot(range(1, len(J_hist) + 1), J_hist)
plt.show()