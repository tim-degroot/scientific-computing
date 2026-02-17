from math import sin, pi
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# ========================================
# H
# ========================================

def Jacobi_Iteration(N = 50, epsilon=1e-5):
    c = np.zeros((N+1,N+1))
    c[0,:] = np.ones(N+1)
    c_new = np.copy(c)
    it = 0
    while 0<1:
        for i in range(N+1):
            for j in range(N+1):
                if j!=0 and j!=N:
                    c_new[j][i] = (1/4) * (c[j][(i+1) % (N+1)] + c[j][(i-1) % (N+1)] + c[j+1][i] + c[j-1][i])
        it += 1
        delta = np.absolute(c_new - c)
        if np.max(delta) < epsilon:
            break
        c = np.copy(c_new)
    return c_new.round(2), it


def SOR_Iteration(omega, N = 50, epsilon=1e-5):
    c = np.zeros((N+1,N+1))
    c[0,:] = np.ones(N+1)
    c_new = np.copy(c)
    it = 0
    while True:
        for i in range(N+1):
            for j in range(1,N):
                # if j!=0 and j!=N:
                c_new[j][i] = ((omega/4)*(c[j][(i+1) % (N+1)] + c_new[j][(i-1) % (N+1)] + c[j+1][i] + c_new[j-1][i]) + (1-omega)*c[j][i])
        it += 1
        delta = np.absolute(c_new - c)
        if np.max(delta) < epsilon:
            return c_new.round(2), it
        c = np.copy(c_new)
    return c_new.round(2), it

# print(Jacobi_Iteration(N=50))
# print(SOR_Iteration(omega=1, N=50)) # Gauss_Seidel
# print(SOR_Iteration(N=50))


# ========================================
# I
# ========================================

p_vals = list(range(11))
omega_vals = [.5, 1.25, 1.5, 1.6, 1.7, 1.8, 1.9]
it_vals_jac = []
it_vals_GS = []
it_vals_SOR = [[], [], [], [], [], [], []]

# for p in p_vals:
#     c_jac, it_jac = Jacobi_Iteration(N=50, epsilon=10*np.e**(-p))
#     c_GS, it_GS = SOR_Iteration(omega=1, N=50, epsilon=10*np.e**(-p))
#     for w in range(len(omega_vals)):
#         c_SOR, it_SOR = SOR_Iteration(omega=omega_vals[w], N=50, epsilon=10*np.e**(-p))
#         it_vals_SOR[w].append(it_SOR)

#     it_vals_jac.append(it_jac)
#     it_vals_GS.append(it_GS)

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.semilogy(p_vals, it_vals_jac, marker='o', color='r', label='Jacobian')
# ax1.semilogy(p_vals, it_vals_GS, marker='o', color='b', label='Gauss-Seidel')
# ax1.semilogy(p_vals, it_vals_SOR[-2], marker='o', color='g', label=r'SOR, $\omega$=1.8')
# for w in range(len(omega_vals)):
#     ax2.semilogy(p_vals, it_vals_SOR[w], marker='o', label=r'$\omega$={}'.format(omega_vals[w]))

# ax1.legend()
# ax2.legend(title=r"SOR $\omega$")
# plt.show()


# ========================================
# J
# ========================================

def SOR_Optimize(omega):
    # omega = float(omega)
    _, it = SOR_Iteration(omega[-1])
    return it

op_omega = sp.optimize.minimize(SOR_Optimize, x0=1.8, bounds=[(1.7, 2.0)])
print(op_omega)

# omegas = np.linspace(1.7, 2.0-1e-6, 10)
# iters = [SOR_Optimize([w]) for w in omegas]

# best_idx = np.argmin(iters)
# best_omega = omegas[best_idx]
# best_iters = iters[best_idx]
# print(best_omega, best_iters)