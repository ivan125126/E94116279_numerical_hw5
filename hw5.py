import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def exact_y(t):
    return t * np.tan(np.log(t))

def f1(t, y):
    return 1 + (y / t) + (y / t)**2

def df1(t, y):
    return (-y/t**2) - (2 * y**2 / t**3) + (1/t + 2*y/t**2) * f1(t, y)

# Euler 方法
def euler_method(f, t0, y0, h, t_end):
    t = np.arange(t0, t_end+h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1])
    return t, y

# Taylor 2階 方法
def taylor2_method(f, df, t0, y0, h, t_end):
    t = np.arange(t0, t_end+h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1]) + (h**2 / 2) * df(t[i-1], y[i-1])
    return t, y
h = 0.1
t0 = 1
y0 = 0
t_end = 2

# Euler
t_euler, y_euler = euler_method(f1, t0, y0, h, t_end)

# Taylor
t_taylor, y_taylor = taylor2_method(f1, df1, t0, y0, h, t_end)

# Exact
y_exact_euler = exact_y(t_euler)
y_exact_taylor = exact_y(t_taylor)

# ====== 表格 (Euler) ======
data_euler = {
    't': t_euler,
    'Euler y': y_euler,
    'Exact y': y_exact_euler,
    'Error': np.abs(y_euler - y_exact_euler)
}
df_euler = pd.DataFrame(data_euler)
print("\n=== Euler Method 結果 ===")
print(df_euler)

# ====== 表格 (Taylor) ======
data_taylor = {
    't': t_taylor,
    'Taylor y': y_taylor,
    'Exact y': y_exact_taylor,
    'Error': np.abs(y_taylor - y_exact_taylor)
}
df_taylor = pd.DataFrame(data_taylor)
print("\n=== Taylor 2nd Order Method 結果 ===")
print(df_taylor)

# ====== 畫圖 (Euler) ======
plt.figure(figsize=(8,6))
plt.plot(t_euler, y_exact_euler, label='Exact Solution', color='black')
plt.plot(t_euler, y_euler, 'o-', label='Euler Method', color='red')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Problem 1 (a): Euler Method vs Exact Solution')
plt.legend()
plt.grid(True)
plt.show()

# ====== 畫圖 (Taylor) ======
plt.figure(figsize=(8,6))
plt.plot(t_taylor, y_exact_taylor, label='Exact Solution', color='black')
plt.plot(t_taylor, y_taylor, 's-', label='Taylor 2nd Order', color='blue')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Problem 1 (b): Taylor 2nd Order vs Exact Solution')
plt.legend()
plt.grid(True)
plt.show()
