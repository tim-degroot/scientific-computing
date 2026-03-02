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
        self.n_timesteps = 10000  # Number of timesteps (convergence around 10k/20k)
    
    
    # def diffusion(self, c):
    #     '''
    #     Discretized two-dimensional diffusion with periodic boundaries along both directions. 
        
    #     Input: 
    #     - c: concentration (NxN np.array)
    #     - i: index of x
    #     - j: index of y 
        
    #     Returns:
    #     - c_dif: diffused concentration at (i,j)
    #     '''
        
    #     # Get N
    #     N = c.shape[0]
        
    #     # Conpute the concentration for t+1 at [i,j]
    #     # (i+-1)%N for periodic boundaries x and y
    #     c_dif =  (c[(i+1)%(N), j] + c[(i-1)%(N), j] + c[i, (j+1)%(N)] + c[i, (j-1)%(N)] - 4*c[i, j]) / self.dx**2
        
    #     return c_dif
    
    def diffusion(self, c):
        # This calculates: (left + right + up + down - 4*center)/dx^2
        return (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) + np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 4 * c) / self.dx**2
    
    
    # def u_prime(self, u, v):
    #     '''
    #     Compute the concentrations u for x and y at time k+1 based on concentrations at time k.
    #     '''

    #     # Get number of intervals
    #     N = u.shape[0]
        
    #     # Create template for k+1
    #     next_u = np.zeros_like(u)
        
    #     # Run nested for loop for x (i) and y (j)
    #     for j in range(N):
    #         for i in range(N):
    #                 next_u[i,j] = u[i,j] + \
    #                     (self.Du * self.diffusion(u, i, j) - u[i,j] * v[i,j]**2 + self.feed * (1- u[i,j])) * self.dt
        
    #     return next_u

    def u_prime(self, u, v):
        '''
        Compute the concentrations u for x and y at time k+1 based on concentrations at time k.
        '''

        # The vectorized update 
        def_u = self.diffusion(u)
        next_u = u + (self.Du * def_u + u * v**2 - self.feed * (1 - u)) * self.dt
        
        return next_u
    

    def v_prime(self, v, u):
        
        lap_v = self.diffusion(v)
        next_v = v + (self.Dv * lap_v + u * v**2 - v * (self.feed + self.kill)) * self.dt
        
        return next_v

    # def v_prime(self, v, u):
    
    #     # Get number of intervals
    #     N = v.shape[0]
        
    #     # Create template for k+1
    #     next_v = np.zeros_like(v)
        
    #     # Run nested for loop for x (i) and y (j)
    #     for j in range(N):
    #         for i in range(N):
    #                 next_v[i,j] = v[i,j] + \
    #                     (self.Dv * self.diffusion(v, i, j) + u[i,j] * v[i,j]**2 - v[i,j]*(self.feed+self.kill)) * self.dt
                    
    #     return next_v
    
    def run_gray_scott(self, u_init, v_init):
          
        # Inital conditions
        u = u_init
        v = v_init
        
        # # Get number of intervals
        # N = u.shape[0]
        
        # # allocate array and put initial conditon at n=0
        # saved_u = np.zeros((N, N, self.n_timesteps+1))
        # saved_u[:,:,0] = u
        # saved_v = np.zeros((N, N, self.n_timesteps+1))
        # saved_v[:,:,0] = v
        
        # RUn the Gray Scott model for n_timesteps
        for i in range(1, self.n_timesteps+1):
            # Determine u at t + 1
            u = self.u_prime(u, v)
            # saved_u[:,:,i] = u
            
            # Determine v at t + 1
            v = self.v_prime(v, u)
            # saved_v[:,:,i] = v
            
            # u = np.clip(u, 0, 1)
            # v = np.clip(v, 0, 1)
        
        return u, v
    
    
if __name__=='__main__':
    
    # Define parameters
    args = {"dt": 1, "dx": 2.5/256, "Du": 0.00002, "Dv": 0.00001,  'feed': 0.02, 'kill': 0.05}
    
    # Intervals
    N = 256
    
    noise = 0.01
    
    
    
    # Intial condition
    u_init = np.ones((N,N)) * 0.5 * (1 + (np.random.random() * 2 - 1) * noise)


    square_size = 5
    v_init = np.zeros((N,N))
    v_init[N//2 - square_size: N//2 + square_size, N//2 - square_size: N//2 + square_size] += 0.25 # Small square in middel
    v_init *= (1 + (np.random.random() * 2 - 1) * noise)
    
    # Create Gray Scott model with given parameters
    reaction_diffusion = GrayScott(**args)
    
    # Run model with initial conditions
    u, v = reaction_diffusion.run_gray_scott(u_init, v_init)
    
    heat = plt.imshow(u, vmin=0, vmax=1)
    plt.colorbar(heat)
    plt.show()
    
    heat = plt.imshow(v,vmin=0, vmax=1)
    plt.colorbar(heat)
    plt.show()

