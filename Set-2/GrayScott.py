# Import functions
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc
from matplotlib.animation import FuncAnimation

class GrayScott():
    def __init__(self, dt, Du, Dv, feed, k, u_0, v_0):
        self.dt = dt
        self.Du = Du
        self.Dv = Dv
        self.feed = feed
        self.k = k
        self.u_0 = u_0
        self.v_0 = v_0
    
    
    def diffusion(self, c, i, j):
        # Get N
        N = c.shape[0]
        
        # Conpute the concentration for t+1 at [i,j]
        # (i+-1)%N for periodic boundaries x and y
        c = c[i,j] + (self.dt*self.D) / (self.dx**2) *\
            (c[(i+1)%(N), j] + c[(i-1)%(N), j] + c[i, (j+1)%(N)] + c[i, (j-1)%(N)] - 4*c[i, j])
        
        return c
    
    @njit
    def u_prime(self, u, v):
        '''
        Compute the concentrations c for x and y at time k+1 based on concentrations at time k.
        '''
    
        N = u.shape[0]
        
        # Create template for k+1
        next_u = u.copy()
        
        # Run nested for loop for x (i) and y (j)
        for j in range(N):
            for i in range(N):
                    next_u[i,j] = u[i,j] + (self.diffusion(u, i, j) - u[i,j] * v[i,j]**2 + self.feed * (1- u[i,j]))
        
        return next_u
    
    @njit
    def v_prime(self, v, u):
    
        N = v.shape[0]
        
        # Create template for k+1
        next_v = v.copy()
        
        # Run nested for loop for x (i) and y (j)
        for j in range(N):
            for i in range(N):
                    next_v[i,j] = v[i,j] + (self.diffusion(v, i, j) + u[i,j] * v[i,j]**2 - v[i,j]*(self.feed+self.k))
                    
        return next_v
    
    def run_gray_scott(self, u_init, v_init):
        
        u = u_init
        v = v_init
        
        saved_u = saved_v = np.zeros(self.N, self.N, self.n_timesteps)
        
        
        for i in range(self.n_timesteps):
                saved_u[i] = self.u_prime(u, v).copy()
                saved_v[i] = self.v_prime(v, u).copy()
        
        
        return saved_u, saved_v
    
    
if __name__=='__main__':
    
    args = {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08,  'feed': 0.035, 'k': 0.060}
    u_init = ...
    v_init = ...
    
    
    reaction_diffusion = GrayScott(**args)
    reaction_diffusion.run_gray_scott(u_init, v_init)