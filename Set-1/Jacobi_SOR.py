# import functions
import numpy as np
import matplotlib.pyplot as plt

class TimeIndependentDiffusion():
    def __init__(self, N, omega, epsilon, max_iter):
        self.N = N
        self.omega = omega
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.c_init = np.zeros((N+1,N+1))
        self.c_init[0,:] = np.ones(N+1)

    
    def Jacobi_Iteration(self, epsilon):
        '''
        Find steady state using Jacobi iteration
        '''
        c = np.copy(self.c_init)
        c_new = np.copy(self.c_init)
        it = 0
        while it < self.max_iter:
            for i in range(self.N + 1):
                for j in range(1, self.N):
                    c_new[j][i] = (1/4) * (c[j][(i+1) % (self.N + 1)] + c[j][(i-1) % (self.N + 1)] + c[j+1][i] + c[j-1][i])
            it += 1
            delta = np.absolute(c_new - c)
            if np.max(delta) < epsilon:
                break
            c = np.copy(c_new)
        return c_new.round(2), it


    def SOR_Iteration(self, omega, epsilon, objects=[], insulation=None):
        '''
        Find steady state using SOR method
        '''

        c = np.copy(self.c_init)
        c_new = np.copy(self.c_init)

        # Create a mask for objects
        c_obj = np.ones((self.N + 1,self.N +1))
        if insulation == None:
            insulation = np.ones(len(objects))
        for n, l in enumerate(objects):
            i,j,k = l[0],l[1],l[2]
            obj = (1 - insulation[n])*np.ones((k,k))
            c_obj[i:i+k, j:j+k] = obj

        it = 0
        while it<self.max_iter:
            for i in range(self.N + 1):
                for j in range(1, self.N):
                    if c_obj[j][i] != 0:
                        c_new[j][i] = (omega/4)*(c[j][(i+1) % (self.N + 1)] + c_new[j][(i-1) % (self.N + 1)] + c[j+1][i] + c_new[j-1][i]) + (1-omega)*c[j][i]
                        c_new[j][i] *= c_obj[j][i]
            it += 1
            delta = np.absolute(c_new - c)
            if np.max(delta) < epsilon:
                return c_new.round(2), it
            c = np.copy(c_new)
        return c_new.round(2), it


    def compare_methods(self, filename, show):
        '''
        Compare the amount of iterations needed to converge for different methods, depending on stopping condition
        '''

        p_vals = list(range(6))
        omega_vals = [.5, 1.5, 1.7, 1.8, 1.9]
        it_vals_jac = []
        it_vals_GS = []
        it_vals_SOR = [[], [], [], [], []]

        for p in p_vals:
            c_jac, it_jac = self.Jacobi_Iteration(epsilon=10**(-p))
            c_GS, it_GS = self.SOR_Iteration(omega=1, epsilon=10**(-p))
            for w in range(len(omega_vals)):
                c_SOR, it_SOR = self.SOR_Iteration(omega=omega_vals[w], epsilon=10**(-p))
                it_vals_SOR[w].append(it_SOR)

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
        
        if filename is not None:
            plt.savefig(filename, dpi=300)

        if show:
            plt.show()

        plt.close(fig)


    def compare_object_location(self, filename, show):
        '''
        Find the effect of object location on convergence time
        '''

        fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(6,6), sharey=True)

        c_0, it_0 = self.SOR_Iteration(self.omega, epsilon=self.epsilon)
        print('no objects:', it_0)
        first = ax1.imshow(c_0, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax1.set_title('no objects\n iterations = {}'.format(it_0))

        c_1, it_1 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[5,20,10]])
        print('1 object, high:', it_1)
        ax2.imshow(c_1, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax2.set_title('1 object, high\n iterations = {}'.format(it_1))

        c_2, it_2 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[20,20,10]])
        print('1 object, middle:', it_2)
        ax3.imshow(c_2, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax3.set_title('1 object, middle\n iterations = {}'.format(it_2))

        c_3, it_3 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[35,20,10]])
        print('1 object, low:', it_3)
        ax4.imshow(c_3, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax4.set_title('1 object, low \n iterations = {}'.format(it_3))

        fig.supxlabel('X')
        fig.supylabel('Y')
        fig.colorbar(first, ax=(ax1,ax2,ax3,ax4))

        if filename is not None:
            plt.savefig(filename, dpi=300)

        if show:
            plt.show()

        plt.close(fig)


    def compare_multiple_objects(self, filename, show):
        '''
        Find the effect of object configuration on convergence time
        '''

        fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(6,6), sharey=True)

        c_7, it_7 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[20,10,10], [20,30,10]])
        print('2 objects, horz:', it_7)
        first = ax1.imshow(c_7, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax1.set_title('2 objects, horizontal\n iterations = {}'.format(it_7))

        c_8, it_8 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[10,20,10], [30,20,10]])
        print('2 objects, vert:', it_8)
        ax2.imshow(c_8, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax2.set_title('2 objects, vertical\n iterations = {}'.format(it_8))

        c_9, it_9 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[10,10,10], [30,30,10]])
        print('2 objects, diag:', it_9)
        ax3.imshow(c_9, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax3.set_title('2 objects, diagonal\n iterations = {}'.format(it_9))

        c_10, it_10 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[10,10,10], [30,10,10], [10,30,10], [30,30,10]])
        print('4 objects, sqr:', it_10)
        ax4.imshow(c_10, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax4.set_title('4 objects, square\n iterations = {}'.format(it_10))

        fig.supxlabel('X')
        fig.supylabel('Y')
        fig.colorbar(first, ax=(ax1,ax2, ax3,ax4))
        
        if filename is not None:
            plt.savefig(filename, dpi=300)

        if show:
            plt.show()

        plt.close(fig)


    def compare_object_insulation(self, filename, show):
        '''
        Find the effect of object insulation level on convergence time
        '''

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6,6), sharey=True)

        c_3, it_3 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[20,20,10]], insulation=[0.5])
        print('insulation = 0.5:', it_3)
        first = ax1.imshow(c_3, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax1.set_title('insulation = 0.5\n iterations = {}'.format(it_3))

        c_4, it_4 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[20,20,10]], insulation=[0.25])
        print('insulation = 0.25:', it_4)
        ax2.imshow(c_4, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax2.set_title('insulation = 0.25\n iterations = {}'.format(it_4))

        c_5, it_5 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[20,20,10]], insulation=[0.1])
        print('insulation = 0.1:', it_5)
        ax3.imshow(c_5, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax3.set_title('insulation = 0.1\n iterations = {}'.format(it_5))

        c_6, it_6 = self.SOR_Iteration(self.omega, epsilon=self.epsilon, objects=[[20,20,10]], insulation=[0.01])
        print('insulation = 0.01:', it_6)
        ax4.imshow(c_6, extent=[0, 1, 0, 1], cmap='plasma', interpolation='nearest')
        ax4.set_title('insulation = 0.01\n iterations = {}'.format(it_6))

        fig.supxlabel('X')
        fig.supylabel('Y')
        fig.colorbar(first, ax=(ax1,ax2, ax3,ax4))
        
        if filename is not None:
            plt.savefig(filename, dpi=300)

        if show:
            plt.show()

        plt.close(fig)


    def optimize_omega(self, include_objects):
        '''
        Find the optimal omega for SOR method, with and without objects
        '''

        def SOR_Optimize(omega, objects=[]):
            _, it = self.SOR_Iteration(omega[-1], epsilon=self.epsilon, objects=objects)
            return it
        
        omegas = np.linspace(1.7, 2.0, 101)
        if self.N == 10:
            omegas = np.linspace(1.5, 1.8, 101)
        omegas = omegas[:-1]
        iters = [np.inf]
        for w in omegas:
            it = SOR_Optimize([w])
            if it > iters[-1]:
                break
            iters.append(it)
        iters = iters[1:]

        best_idx = np.argmin(iters)
        best_omega = omegas[best_idx]
        best_iters = iters[best_idx]
        print('Best omega:', best_omega, ', with', best_iters, 'iterations')

        if include_objects:
            for obj in [[5,20,10], [20,20,10], [35,20,10]]:
                iters_obj = [np.inf]
                for w in omegas:
                    it = SOR_Optimize([w], objects=[obj])
                    if it > iters_obj[-1]:
                        break
                    iters_obj.append(it)
                iters_obj = iters_obj[1:]
                
                best_idx_obj = np.argmin(iters_obj)
                best_omega_obj = omegas[best_idx_obj]
                best_iters_obj = iters_obj[best_idx_obj]

                print('For object ', obj)
                print('Best omega:', best_omega_obj, ', with', best_iters_obj, 'iterations')


if __name__ == '__main__':
    args1 = {"N": 50, "omega": 1.9, "epsilon": 1e-5, "max_iter": 10000}
    args2 = {"N": 25, "omega": 1.9, "epsilon": 1e-5, "max_iter": 10000}
    args3 = {"N": 10, "omega": 1.9, "epsilon": 1e-5, "max_iter": 10000}
    model1 = TimeIndependentDiffusion(**args1)
    model2 = TimeIndependentDiffusion(**args2)
    model3 = TimeIndependentDiffusion(**args3)
    print('Comparing iteration methods')
    model1.compare_methods(filename='Jac_SOR', show=True)
    print('Comparing object location:')
    model1.compare_object_location(filename='object_location', show=True)
    print('comparing multiple object formations:')
    model1.compare_multiple_objects(filename='multiple_objects', show=True)
    print('Comparing object insulation levels:')
    model1.compare_object_insulation(filename='object_insulation', show=True)
    print('Finding optimal omega(s):')
    model1.optimize_omega(include_objects=True)
    model2.optimize_omega(include_objects=False)
    model3.optimize_omega(include_objects=False)