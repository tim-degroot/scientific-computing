# Import functions
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc
from matplotlib.animation import FuncAnimation

class TimeDepententDiffusion():
    def __init__(self, N, D, dt, data_res):
        
        self.N = N
        self.D = D
        self.dx = 1/N
        self.dt = dt
        self.data_res = data_res
        self.n_timesteps = int(1/dt)
        self.c_init = np.zeros((N, N))
        self.c_init[:,N-1] = 1
        
        # Assert stability
        assert (4*self.dt*self.D)/(self.dx**2) <= 1
        
        @njit
        def single_diffusion(c, dx, dt, D):
            '''
            Compute the concentrations c at time k+1 based on concentrations at the current time
            '''
            
            X, Y = c.shape
            
            # Create template for k+1
            next_c = np.zeros((X,Y))
            
            # Run nested for loop for x and y
            for i in range(X):
                for j in range(Y):
                    
                    # Boundry conditions for Y
                    if j == 0:
                        next_c[i,j] = 0  
                    elif j == Y-1:
                        next_c[i,j] = 1
                    
                    # Boundry conditions for X
                    elif i == 0:
                        next_c[i,j] = c[i,j] + (dt*D) / (dx**2) * (c[i+1, j] + c[X-1, j] + c[i, j+1] + c[i, j-1] - 4*c[i, j])
                    elif i == X-1:
                        next_c[i,j] = c[i,j] + (dt*D) / (dx**2) * (c[1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4*c[i, j])
                    
                    # Outside of boundry
                    else:
                        next_c[i,j] = c[i,j] + (dt*D) / (dx**2) * (c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4*c[i, j])
            
            return next_c

        def run_diffusion(c_init, n_timesteps, dx, dt, D, data_res):
            '''
            Runs the diffusion model for n_timesteps 
            '''
            c = c_init
            
            # allocate matrix to store data
            x, y = c.shape
            n_to_save = int(round(1/data_res) + 1)
            c_data = np.zeros((x, y, n_to_save))
            save_index = 0
            
            # Run loop the number of timesteps (+1 to also save t=0)
            for k in range(n_timesteps+1):
                
                # Compute concentrations for k+1
                c = single_diffusion(c, dx, dt, D)
                
                # Save data so time resolution of saved data corresponds with data_res
                if k%(data_res*n_timesteps) == 0:
                    
                    # Save concentrations of k+1 in c[x,y,k] matrix
                    c_data[:,:,save_index] = c.copy()
                    save_index += 1
                
            return c_data
    
        self.c_data = run_diffusion(self.c_init, self.n_timesteps, self.dx, self.dt, self.D, self.data_res)

    def heatmap(self, filename, show):
        
        times = [0.001, 0.01, 0.1, 1]
        
        # How to plot the subplots
        axes = [(0,0), (0,1), (1,0), (1,1)]
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(6,6), sharey=True, sharex=True)

        # Plot a heatmap of the concentration as a function of x and y given timestep t
        for i, t in enumerate(times):
            
            heatmap = axs[axes[i]].pcolor(self.c_data[:,:,int(t/self.data_res)], cmap='plasma')
            axs[axes[i]].set_title(f't = {t}')

        fig.supxlabel('X')
        fig.supylabel('Y')

        plt.colorbar(heatmap)
        plt.tight_layout()
        
        if filename is not None:
            plt.savefig(filename, dpi=300)

        if show:
            plt.show()

        plt.clf()
        
    def compare_analytical(self, filename, show):
        '''
        Makes a 2x2 plot showing the time-dependent 2D diffusion model approximation against the analytical solution 
        at various timesteps.
        '''
        
        def analytical_diffusion(x, t, D, n_terms=10):

            '''
            Solve the analytical solution for the 2D Time-dependent diffusion model with:
                Boundry conditions:
                    0 <= x,y <= 1
                    c(x,y=0,t) = 0; c(x,y=1,t) = 1
                    x has a perodic boundry
                Initial condition:
                    c(x,y<1,0) = 0
            '''

            c = np.zeros_like(x)
            
            for i in range(n_terms):
                term_1 = erfc((1-x+2*i)/(2*np.sqrt(D*t)))
                term_2 = erfc((1+x+2*i)/(2*np.sqrt(D*t)))
                c += term_1 - term_2
            
            return c
        
        # Time steps to compare
        time = [0.001, 0.01, 0.1, 1]
        axes = [(0,0), (0,1), (1,0), (1,1)]

        # Create interactive plot
        fig, axs = plt.subplots(2, 2, figsize=(6,6), sharey=True, sharex=True)
        fig.supxlabel('X')
        fig.supylabel('Y')
        
        x = np.linspace(0, 1, self.N)

        for i, t in enumerate(time):
            axs[axes[i]].plot(x, self.c_data[0,:,int(t/self.data_res)])
            axs[axes[i]].plot(x, analytical_diffusion(x, t, self.D))
            axs[axes[i]].set_title(f't = {t}')

        plt.tight_layout()
        
        # Save plot
        if filename is not None:
            plt.savefig(filename, dpi=300)

        # Show plot
        if show:
            plt.show()

        plt.clf()    
    
    def animate(self, interval=30, filename=None, show=True):
        '''
        animates the 2 dimensional diffusion model over time
        '''
                
        fig, ax = plt.subplots()
        heat = ax.imshow(self.c_data[:,:,0], cmap='plasma')
        ax.set(xlabel='X', ylabel='Y')

        def update(frame):
            heat.set_data(self.c_data[:,:,frame])
            return (heat,)


        anim = FuncAnimation(fig, update, frames=int(np.round(1/self.data_res)/2), interval=interval)

        # Save animation
        if filename is not None:
            anim.save(filename)

        # Show animation
        if show:
            plt.show()
        
        plt.clf()

if __name__ == '__main__':
    args = {"N": 100, "D": 1, "dt": 0.000025, "data_res": 0.001}
    model = TimeDepententDiffusion(**args)
    model.heatmap("diffusion_heatmap.jpeg", show=True)
    model.compare_analytical("compare_analytical.jpeg", show=True)
    model.animate()

