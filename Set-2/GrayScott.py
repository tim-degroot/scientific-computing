# Import functions
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc
from matplotlib.animation import FuncAnimation

class GrayScott():
    def __init__(self, dt, dx, Du, Dv, feed, kill):
        self.dt = dt
        self.dx = dx
        self.Du = Du
        self.Dv = Dv
        self.feed = feed
        self.kill = kill
        self.n_timesteps = 500
    
    
    def diffusion(self, c, i, j):
        # Get N
        N = c.shape[0]
        
        # Conpute the concentration for t+1 at [i,j]
        # (i+-1)%N for periodic boundaries x and y
        c =  (c[(i+1)%(N), j] + c[(i-1)%(N), j] + c[i, (j+1)%(N)] + c[i, (j-1)%(N)] - 4*c[i, j]) / self.dx**2
        
        return c
    

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
                    next_u[i,j] = u[i,j] + (self.Du * self.diffusion(u, i, j) - u[i,j] * v[i,j]**2 + self.feed * (1- u[i,j])) * self.dt
        
        return next_u
    

    def v_prime(self, v, u):
    
        N = v.shape[0]
        
        # Create template for k+1
        next_v = v.copy()
        
        # Run nested for loop for x (i) and y (j)
        for j in range(N):
            for i in range(N):
                    next_v[i,j] = v[i,j] + (self.Dv * self.diffusion(v, i, j) + u[i,j] * v[i,j]**2 - v[i,j]*(self.feed+self.kill)) * self.dt
                    
        return next_v
    
    def run_gray_scott(self, u_init, v_init):
          
        # Inital conditions
        u = u_init
        v = v_init
        
        # Get number of intervals
        N = u.shape[0]
        
        # allocate array and put initial conditon at n=0
        saved_u = np.zeros((N, N, self.n_timesteps+1))
        saved_u[:,:,0] = u
        saved_v = np.zeros((N, N, self.n_timesteps+1))
        saved_v[:,:,0] = v
        
        # RUn the Gray Scott model for n_timesteps
        for i in range(1, self.n_timesteps+1):
                
                # Determine u at t + 1
                u = self.u_prime(u, v).copy()
                saved_u[:,:,i] = u
                
                # Determine v at t + 1
                v = self.v_prime(v, u).copy()
                saved_v[:,:,i] = v
        
        return saved_u, saved_v
    
    
if __name__=='__main__':
    
    # Define parameters
    args = {"dt": 1, "dx": 1, "Du": 0.16, "Dv": 0.08,  'feed': 0.035, 'kill': 0.060}
    
    # Intervals
    N = 100
    
    # Intial condition
    u_init = np.ones((N,N)) * 0.5

    v_init = np.zeros((N,N))
    v_init[N//2 - 2: N//2 + 2, N//2 - 2: N//2 + 2] = 0.25 # Small square in middel
    
    # Create Gray Scott model with given parameters
    reaction_diffusion = GrayScott(**args)
    
    # Run model with initial conditions
    u_data, v_data = reaction_diffusion.run_gray_scott(u_init, v_init)
        
    # Plot the system
    heat = plt.imshow(v_data[:,:,-1])
    plt.show()
    
    heat = plt.imshow(u_data[:,:,-1])
    plt.show()

