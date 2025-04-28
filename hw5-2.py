import math
import sys

try:
    import pandas as pd
except ImportError:
    print("請先安裝 pandas")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("請先安裝 matplotlib")
    sys.exit(1)

def f_system(t, u1, u2):
    du1 = round(9 * u1 + 24 * u2 + 5 * math.cos(t) - 1/3 * math.sin(t), 8)
    du2 = round(-24 * u1 - 52 * u2 - 9 * math.cos(t) + 1/3 * math.sin(t), 8)
    return du1, du2

def u_exact(t0, h, n):
    ti = [round(t0 + i * h, 8) for i in range(n + 1)]
    u1 = [round(2 * math.exp(-3*t) - math.exp(-39*t) + 1/3 * math.cos(t), 8) for t in ti]
    u2 = [round(-math.exp(-3*t) + 2 * math.exp(-39*t) - 1/3 * math.cos(t), 8) for t in ti]
    return u1, u2

def RK4_system(t, u10, u20, h, n):
    ti = [round(t + i * h, 8) for i in range(n + 1)]
    u1i = [0] * (n + 1)
    u2i = [0] * (n + 1)
    u1i[0] = u10
    u2i[0] = u20
    for i in range(n):
        k1_1, k1_2 = f_system(ti[i], u1i[i], u2i[i])
        k2_1, k2_2 = f_system(ti[i] + h/2, u1i[i] + h/2 * k1_1, u2i[i] + h/2 * k1_2)
        k3_1, k3_2 = f_system(ti[i] + h/2, u1i[i] + h/2 * k2_1, u2i[i] + h/2 * k2_2)
        k4_1, k4_2 = f_system(ti[i] + h, u1i[i] + h * k3_1, u2i[i] + h * k3_2)
        u1i[i+1] = round(u1i[i] + h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1), 8)
        u2i[i+1] = round(u2i[i] + h/6 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2), 8)
    return ti, u1i, u2i

t0 = 0
t_end = 1
u10 = 4/3
u20 = 2/3

for h in [0.05, 0.1]:
    n = int((t_end - t0) / h)
    ti, u1i_rk4, u2i_rk4 = RK4_system(t0, u10, u20, h, n)
    u1_exact, u2_exact = u_exact(t0, h, n)

    
    err_u1 = [round(abs(u1e - u1r), 8) for u1e, u1r in zip(u1_exact, u1i_rk4)]
    err_u2 = [round(abs(u2e - u2r), 8) for u2e, u2r in zip(u2_exact, u2i_rk4)]

    
    df = pd.DataFrame({
        't':        ti,
        'u1_exact': u1_exact,
        'u1_RK4':   u1i_rk4,
        'err_u1':   err_u1,
        'u2_exact': u2_exact,
        'u2_RK4':   u2i_rk4,
        'err_u2':   err_u2
    })

    pd.set_option('display.float_format', lambda x: f'{x:.8f}')
    print(f"\nRK4 Method (h = {h})")
    print(df.to_string(index=False))

    # 畫圖 u1
    plt.figure()
    plt.plot(ti, u1_exact, '-', label='u1 Exact')
    plt.plot(ti, u1i_rk4, '--', label=f'u1 RK4 (h={h})')
    plt.xlabel('t')
    plt.ylabel('u1')
    plt.title(f'u1 vs RK4 (h={h})')
    plt.legend()
    plt.grid(True)

    # 畫圖 u2
    plt.figure()
    plt.plot(ti, u2_exact, '-', label='u2 Exact')
    plt.plot(ti, u2i_rk4, '--', label=f'u2 RK4 (h={h})')
    plt.xlabel('t')
    plt.ylabel('u2')
    plt.title(f'u2 vs RK4 (h={h})')
    plt.legend()
    plt.grid(True)

plt.show()
