# Import functions
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc
from matplotlib.animation import FuncAnimation


class GrayScott:
    def __init__(self, dt, dx, Du, Dv, feed, kill):
        
        self.dt = dt            # Timestep
        self.dx = dx            # Spatial step
        self.Du = Du            # Diffusion constant of u
        self.Dv = Dv            # Diffusion constant of v
        self.feed = feed        # Supply of u (feed)
        self.kill = kill        # Decay parameter (kill)    
    
    def diffusion(self, c):
        # This calculates: (left + right + up + down - 4*center)/dx^2
        return (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) + np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 4 * c) / self.dx**2
    
    
    def u_prime(self, u, v):
        
        lap_u = self.diffusion(u)
        next_u = u + (self.Du*lap_u - u*v*v + self.feed*(1-u)) * self.dt
    
        return next_u

    def v_prime(self, v, u):
        
        lap_v = self.diffusion(v)
        next_v = v + (self.Dv*lap_v + u*v*v - v*(self.feed+self.kill)) * self.dt
        
        return next_v

    
    def run_gray_scott(self, u_init, v_init, n_timesteps):
          
        # Inital conditions
        u = u_init
        v = v_init
        
        # RUn the Gray Scott model for n_timesteps
        for i in range(1, n_timesteps+1):
            # Determine u at t + 1
            u = self.u_prime(u, v)
            
            # Determine v at t + 1
            v = self.v_prime(v, u)
        
        return u, v
    
    def initial_conditions(N, u, v, square_size, noise, seed=42):
    
        rng = np.random.default_rng(seed=seed)
    
        # Initialize u 
        u_init = np.ones((N,N)) * u * (1 + (rng.random((N,N)) * 2 - 1) * noise)

        # Initialize v
        v_init = np.zeros((N,N))
        
        ss = int(np.round(square_size/2))                              # Get half the square size and round it
        
        v_init[N//2 - ss: N//2 + ss, N//2 - ss: N//2 + ss] = v         # Small square in middel
        v_init *= (1 + (rng.random((N,N)) * 2 - 1) * noise)      # Add small amount of noise 
    
        return u_init, v_init  
    
    def plot_argsets(u_init, v_init, args_set):
        
        # Create figure
        fig, axs = plt.subplots(2,2, figsize=(5,4))
        
        axes = [(0,0), (0,1), (1,0), (1,1)]
        
        # Plot a heatmap for the different args
        for i, args in enumerate(args_set):
            
            # Create Gray Scott model with given parameters
            reaction_diffusion = GrayScott(**args)
            
            # Run model with initial conditions for n_timesteps
            u, v = reaction_diffusion.run_gray_scott(u_init, v_init, n_timesteps)
            
            # Plot heatmap
            heatmap = axs[axes[i]].imshow(u, vmin=0, vmax=1, cmap='plasma')
            axs[axes[i]].axis('off')

        plt.tight_layout()
        
        # Make room for colorbar
        plt.subplots_adjust(right=0.85) 

        # Add colorbar
        cbar_ax = fig.add_axes([0.875, 0.15, 0.03, 0.7])
        fig.colorbar(heatmap, ax=axs.ravel().tolist(), aspect=20, cax=cbar_ax)

        plt.show()

    
    
if __name__=='__main__':
    
    # Intervals
    N = 100
    n_timesteps = 40000         
    noise = 0.01            # +-1%
    
    # Get inital conditions for u and v
    u_init, v_init = GrayScott.initial_conditions(N, 0.5, 0.25, 10, 0.01)
    
    # Define parameters to plot
    args_set = [{"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, 'feed': 0.035, 'kill': 0.06},
                {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, 'feed': 0.05, 'kill': 0.065},
                {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, 'feed': 0.015, 'kill': 0.044},
                {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, 'feed': 0.04, 'kill': 0.065}]
    
    GrayScott.plot_argsets(u_init, v_init, args_set)
    
    # args = {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, 'feed': 0.05, 'kill': 0.065}
    
    # # Create Gray Scott model with given parameters
    # reaction_diffusion = GrayScott(**args)
    
    # # Run model with initial conditions for n_timesteps
    # u, v = reaction_diffusion.run_gray_scott(u_init, v_init, n_timesteps)
    
    # # Plot heatmap
    # heatmap = plt.imshow(u, vmin=0, vmax=1, cmap='plasma')
    # plt.show()
    


