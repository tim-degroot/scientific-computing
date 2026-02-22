# Import functions
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc
from matplotlib.animation import FuncAnimation

class TimeDepententDiffusion():
    def __init__(self, N, D, dt, data_res, times):
        
        self.N = N
        self.D = D
        self.dx = 1/N
        self.dt = dt
        self.data_res = data_res
        self.n_timesteps = int(1/dt)
        self.c_init = np.zeros((N, N))
        self.c_init[:,N-1] = 1
        self.times = times
        
        # Assert stability
        assert (4*self.dt*self.D)/(self.dx**2) <= 1
        
        @njit
        def single_diffusion(c, dx, dt, D):
            '''
            Compute the concentrations c for x and y at time k+1 based on concentrations at time k.
            '''
            
            N = c.shape[0]
            
            # Create template for k+1
            next_c = c.copy()
            
            # Run nested for loop for x (i) and y (j)
            # For j==0 and j==N keep precious value (0 and 1 res.)
            for j in range(1, N-1):
                for i in range(N):
                        # (i+-1)%N for periodic boundaries x
                        next_c[i,j] = c[i,j] + (dt*D) / (dx**2) *\
                            (c[(i+1)%(N), j] + c[(i-1)%(N), j] + c[i, j+1] + c[i, j-1] - 4*c[i, j])
            
            return next_c

        def run_diffusion(c_init, n_timesteps, dx, dt, D, data_res):
            '''
            Runs the diffusion model for n_timesteps 
            '''
            c = c_init
            
            # allocate matrix to store data
            N = c.shape[0]
            n_to_save = int(round(1/data_res) + 1)
            c_data = np.zeros((N, N, n_to_save))
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
    
        # How to plot the subplots
        axes = [(0,0), (0,1), (1,0), (1,1)]
        
        # Create figure
        fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
        fig.supxlabel('X')
        fig.supylabel('Y')
        
        # Plot a heatmap of the concentration as a function of x and y given timestep t
        for i, t in enumerate(self.times):
            heatmap = axs[axes[i]].imshow(self.c_data[:,:,int(t/self.data_res)].T[::-1], extent=[0, 1, 0, 1], cmap='plasma')
            axs[axes[i]].set_title(f't = {t}')

        # Add colorbar
        cbar_ax = fig.add_axes([0.875, 0.15, 0.03, 0.7])
        fig.colorbar(heatmap, ax=axs.ravel().tolist(), aspect=20, cax=cbar_ax)
        
        # Make room for colorbar
        plt.subplots_adjust(right=0.85) 
        
        if filename is not None:
            plt.savefig(filename, dpi=300)

        if show:
            plt.show()

        plt.close(fig)
        
    def compare_analytical(self, filename, show):
        '''
        Makes a 2x2 plot showing the time-dependent 2D diffusion model approximation against the analytical solution 
        at various timesteps.
        '''
        
        def analytical_diffusion(x, t, D, n_terms=15):

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

        # Create interactive plot
        fig, axs = plt.subplots(sharey=True, sharex=True)
        plt.xlabel('Y')
        plt.ylabel('error (abs(analytical - numerical))')
        
        y = np.linspace(0, 1, self.N)
        
        # Plot difference between numerical and analyical solution
        for t in self.times:
            err = np.abs(self.c_data[0,:,int(t/self.data_res)]-analytical_diffusion(y, t, self.D))
            axs.plot(y, err, label=f't={t}')
            axs.set_yscale('log')
            axs.set_ylim([10E-15, 0])
        
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        if filename is not None:
            plt.savefig(filename, dpi=300)

        # Show plot
        if show:
            plt.show()

        plt.close(fig)   
    
    def animate(self, filename=None, show=True, interval=5):
        '''
        animates the 2 dimensional diffusion model over time
        '''
        
        # Initialize first frame    
        fig, ax = plt.subplots()
        heat = ax.imshow(self.c_data[:,:,0].T[::-1], extent=[0, 1, 0, 1], cmap='plasma')
        plt.colorbar(heat)
        ax.set(xlabel='X', ylabel='Y')

        def update(frame):
            heat.set_data(self.c_data[:,:,frame].T[::-1])
            ax.set_title(f't={np.round(frame*self.data_res, 3)}')
            return (heat,)

        # Animate the diffusion model until steady state (t=1)
        anim = FuncAnimation(fig, update, frames=int(np.round(1/self.data_res)), interval=interval)

        # Save animation
        if filename is not None:
            anim.save(filename, dpi=300)

        # Show animation
        if show:
            plt.show()
        
        plt.clf()

if __name__ == '__main__':
    args = {"N": 100, "D": 1, "dt": 0.000025, "data_res": 0.001, "times": [0.001, 0.01, 0.1, 1]}
    model = TimeDepententDiffusion(**args)
    # model.heatmap("diffusion_heatmap.jpeg", show=True)
    model.compare_analytical("compare_analytical.jpeg", show=True)
    # model.animate("diffusion_progression.gif", show=True)

