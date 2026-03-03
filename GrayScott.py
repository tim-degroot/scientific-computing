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

    
if __name__=='__main__':
    np.seterr(over='raise')
    # Define parameters
    args = {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08, 'feed': 0.035, 'kill': 0.06}
    
    # Intervals
    N = 256
    n_timesteps = 20000
    noise = 0.01
    
    # Intial condition
    u_init = np.ones((N,N)) * 0.5 * (1 + (np.random.random((N,N)) * 2 - 1) * noise)

    square_size = 5
    v_init = np.zeros((N,N))
    v_init[N//2 - square_size: N//2 + square_size, N//2 - square_size: N//2 + square_size] += 0.25 # Small square in middel
    v_init *= (1 + (np.random.random((N,N)) * 2 - 1) * noise)
    
    # Create Gray Scott model with given parameters
    reaction_diffusion = GrayScott(**args)
    
    # Run model with initial conditions for n_timesteps
    u, v = reaction_diffusion.run_gray_scott(u_init, v_init, n_timesteps)
    
    # Plot heatmap of U
    heat = plt.imshow(u, vmin=0, vmax=1)
    plt.colorbar(heat)
    plt.show()


