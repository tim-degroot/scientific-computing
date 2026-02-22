from math import sin, pi
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# ========================================
# H
# ========================================

def Jacobi_Iteration(N = 50, epsilon=1e-5, max_iter=10000):
    c = np.zeros((N+1,N+1))
    c[0,:] = np.ones(N+1)
    c_new = np.copy(c)
    it = 0
    while it<max_iter:
        for i in range(N+1):
            for j in range(1, N):
                c_new[j][i] = (1/4) * (c[j][(i+1) % (N+1)] + c[j][(i-1) % (N+1)] + c[j+1][i] + c[j-1][i])
        it += 1
        delta = np.absolute(c_new - c)
        if np.max(delta) < epsilon:
            break
        c = np.copy(c_new)
    return c_new.round(2), it


def SOR_Iteration(omega, N = 50, epsilon=1e-5, coor_size=[], insulation=None, max_iter=10000):
    c = np.zeros((N+1,N+1))
    c[0,:] = np.ones(N+1)
    c_new = np.copy(c)

    c_obj = np.ones((51,51))
    if insulation == None:
        insulation = np.zeros(len(coor_size))
    for n, l in enumerate(coor_size):
        i,j,k = l[0],l[1],l[2]
        obj = insulation[n]*np.ones((k,k))
        c_obj[i:i+k, j:j+k] = obj

    it = 0
    while it<max_iter:
        for i in range(N+1):
            for j in range(1,N):
                # if j!=0 and j!=N:
                if c_obj[j][i] != 0:
                    c_new[j][i] = (omega/4)*(c[j][(i+1) % (N+1)] + c_new[j][(i-1) % (N+1)] + c[j+1][i] + c_new[j-1][i]) + (1-omega)*c[j][i]
                    c_new[j][i] *= c_obj[j][i]
        it += 1
        delta = np.absolute(c_new - c)
        if np.max(delta) < epsilon:
            return c_new.round(2), it
        c = np.copy(c_new)
    return c_new.round(2), it

# ========================================
# I
# ========================================

p_vals = list(range(6))
omega_vals = [.5, 1.5, 1.7, 1.8, 1.9] # 1.25, 1.6
it_vals_jac = []
it_vals_GS = []
it_vals_SOR = [[], [], [], [], []]

for p in p_vals:
    print('p',p)
    c_jac, it_jac = Jacobi_Iteration(N=50, epsilon=10**(-p))
    print('jacobi:', it_jac)
    c_GS, it_GS = SOR_Iteration(omega=1, N=50, epsilon=10**(-p))
    print('GS:', it_GS)
    for w in range(len(omega_vals)):
        c_SOR, it_SOR = SOR_Iteration(omega=omega_vals[w], N=50, epsilon=10**(-p))
        it_vals_SOR[w].append(it_SOR)
        print('SOR, {}:'.format(omega_vals[w]), it_SOR)

    it_vals_jac.append(it_jac)
    it_vals_GS.append(it_GS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6), sharey=True)
ax1.semilogy(p_vals, it_vals_jac, marker='o', color='g', label='Jacobian')
ax1.semilogy(p_vals, it_vals_GS, marker='o', color='b', label='Gauss-Seidel')
ax1.semilogy(p_vals, it_vals_SOR[-2], marker='o', color='r', label=r'SOR, $\omega$=1.8')
for w in range(len(omega_vals)):
    ax2.semilogy(p_vals, it_vals_SOR[w], marker='o', label=r'$\omega$={}'.format(omega_vals[w]))

ax1.legend()
ax2.legend(title=r"SOR $\omega$")
fig.supxlabel(r'$p\quad (\epsilon=10^{-p})$')
fig.supylabel('number of iterations')
plt.tight_layout()
plt.show()


# ========================================
# J
# ========================================

def SOR_Optimize(omega, coor_size=[]):
    # omega = float(omega)
    _, it = SOR_Iteration(omega[-1], coor_size=coor_size)
    return it

# op_omega = sp.optimize.minimize(SOR_Optimize, x0=1.8, bounds=[(1.7, 2.0)])
# print(op_omega)

print('Finding optimal omega')
omegas = np.linspace(1.8, 2.0, 100)
omegas = omegas[:-1]
# iters = [SOR_Optimize([w]) for w in omegas]

# best_idx = np.argmin(iters)
# best_omega = omegas[best_idx]
# best_iters = iters[best_idx]
# print(best_omega, best_iters)

# ========================================
# K
# ========================================

fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(6,6), sharey=True)

c_0, it_0 = SOR_Iteration(1.9)
print('no objects:', it_0)
first = ax1.imshow(c_0, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax1.set_title('no objects')

c_1, it_1 = SOR_Iteration(1.9, coor_size=[[5,20,10]])
print('1 object, high:', it_1)
ax2.imshow(c_1, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax2.set_title('1 object, high')

c_2, it_2 = SOR_Iteration(1.9, coor_size=[[20,20,10]])
print('1 object, middle:', it_2)
ax3.imshow(c_2, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax3.set_title('1 object, middle')

c_3, it_3 = SOR_Iteration(1.9, coor_size=[[35,20,10]])
print('1 object, low:', it_3)
ax4.imshow(c_3, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax4.set_title('1 object, low')

fig.supxlabel('X')
fig.supylabel('Y')
fig.colorbar(first, ax=(ax1,ax2,ax3,ax4))
plt.show()

fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(6,6), sharey=True)

c_7, it_7 = SOR_Iteration(1.9, coor_size=[[20,10,10], [20,30,10]])
print('2 objects, horz:', it_7)
first = ax1.imshow(c_7, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax1.set_title('2 objects, horizontal')

c_8, it_8 = SOR_Iteration(1.9, coor_size=[[10,20,10], [30,20,10]])
print('2 objects, vert:', it_8)
ax2.imshow(c_8, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax2.set_title('2 objects, vertical')

c_9, it_9 = SOR_Iteration(1.9, coor_size=[[10,10,10], [30,30,10]])
print('2 objects, diag:', it_9)
ax3.imshow(c_9, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax3.set_title('2 objects, diagonal')

c_10, it_10 = SOR_Iteration(1.9, coor_size=[[10,10,10], [30,10,10], [10,30,10], [30,30,10]])
print('4 objects, sqr:', it_10)
ax4.imshow(c_10, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax4.set_title('4 objects, square')

fig.supxlabel('X')
fig.supylabel('Y')
fig.colorbar(first, ax=(ax1,ax2, ax3,ax4))
plt.show()

# iters_1 = [SOR_Optimize([w], [[20,20,10]]) for w in omegas]
# iters_2 = [SOR_Optimize([w], [[10,10,5], [30,30,5]]) for w in omegas]

# best_idx_1 = np.argmin(iters_1)
# best_omega_1 = omegas[best_idx_1]
# best_iters_1 = iters_1[best_idx_1]

# best_idx_2 = np.argmin(iters_2)
# best_omega_2 = omegas[best_idx_2]
# best_iters_2 = iters_2[best_idx_2]

# print(best_omega_1, best_iters_1)
# print(best_omega_2, best_iters_2)

# ========================================
# J
# ========================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6,6), sharey=True)

c_3, it_3 = SOR_Iteration(1.9, coor_size=[[20,20,10]], insulation=[0.5])
print('insulation = 0.5:', it_3)
first = ax1.imshow(c_3, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax1.set_title('insulation = 0.5')

c_4, it_4 = SOR_Iteration(1.9, coor_size=[[20,20,10]], insulation=[0.75])
print('insulation = 0.75:', it_4)
ax2.imshow(c_4, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax2.set_title('insulation = 0.75')

c_5, it_5 = SOR_Iteration(1.9, coor_size=[[20,20,10]], insulation=[0.9])
print('insulation = 0.9:', it_5)
ax3.imshow(c_5, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax3.set_title('insulation = 0.9')

c_6, it_6 = SOR_Iteration(1.9, coor_size=[[20,20,10]], insulation=[0.99])
print('insulation = 0.99:', it_6)
ax4.imshow(c_6, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
ax4.set_title('insulation = 0.99')

fig.supxlabel('X')
fig.supylabel('Y')
fig.colorbar(first, ax=(ax1,ax2, ax3,ax4))
plt.show()

# c_2, it_2 = SOR_Iteration(1.8, coor_size=[[10,10,5], [30,30,5]], insulation=[0.75, 0.25])
# print(it_2)
# plt.imshow(c_2, cmap='plasma', interpolation='nearest')
# plt.show()