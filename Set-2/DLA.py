import numpy as np
import matplotlib.pyplot as plt

class DLA:
    def __init__(self, size_x: int, size_y: int, nu: float):
        self.size_x = size_x
        self.size_y = size_y
        self.nu = nu
        self.grid = np.zeros((size_y+2, size_x))
        self.grid[0] = np.ones(size_x)
        # includes a emission and absorbtion row at the top and bottom
        self.initial_point = [np.random.choice(size_x), size_y]
        # initial point is in the row above the absorbtion row
        self.object = [self.initial_point]

    def SOR_Iteration(self, omega=1.9, epsilon=1e-5, max_iter=10000):
        '''
        Find steady state using SOR method
        '''

        c = np.copy(self.grid)
        c_new = np.copy(c)

        it = 0
        while it<max_iter:
            for x,y in self.object:
                c[y,x] = 0
            # print(it)
            for j in range(1, self.size_y+1):
                for i in range(self.size_x):
                    if [i,j] not in self.object:
                        c_new[j][i] = (omega/4)*(c[j][(i+1) % self.size_x] 
                                                 + c_new[j][(i-1) % self.size_x]  
                                                 + c[j+1][i] 
                                                 + c_new[j-1][i]) + (1-omega)*c[j][i]
            it += 1
            delta = np.absolute(c_new - c)
            if np.max(delta) < epsilon:
                return c_new.round(2), it
            c = np.copy(c_new)
        return c_new, it
    
    def find_neighbors(self):
        neighbors = []
        for point in self.object:
            x,y = point
            neighbors.append([x, min(y+1, self.size_y)])
            neighbors.append([x, max(y-1, 1)])
            neighbors.append([min(x+1, self.size_x-1), y])
            neighbors.append([max(x-1, 0), y])
        
        # could find double neigbors:
        new_neighbors = []
        for n in neighbors:
            if n not in self.object and n not in new_neighbors:
                new_neighbors.append(n)
        return new_neighbors
    
    def one_growth_step(self):
        neighbors = self.find_neighbors()
        print('neighbors are: ',neighbors)
        n_index = np.arange(len(neighbors))
        c, it = self.SOR_Iteration()

        probs = []
        for point in neighbors:
            x, y = point
            prob = c[y,x]**self.nu / (np.sum([c[n[1], n[0]]**self.nu for n in neighbors]))
            probs.append(prob)
        
        print('probabilities are: ', probs)

        random_index = np.random.choice(n_index, p=probs)
        new_object = neighbors[random_index]
        self.object.append(new_object)
        print('new object: ', new_object)
        return c, new_object
    
    def simulate_growth_model(self, steps):
        for t in range(steps):
            print(t)
            c, new_point = self.one_growth_step()
        return c, self.object
    
    def visualize_DLA(self, steps, show_diffusion=False, show_object=False, show_both=True):
        c, object = self.simulate_growth_model(steps=steps)
        c_final, it = self.SOR_Iteration()
        object_matrix = np.zeros_like(c)
        for x,y in object:
            object_matrix[y,x] = 1

        if show_diffusion:
            plt.imshow(c_final, cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.show()
        
        if show_object:
            object_matrix_copy = np.copy(object_matrix)
            object_matrix_copy = np.delete(object_matrix_copy, 0, 0)
            object_matrix_copy = np.delete(object_matrix_copy, -1, 0)
            
            plt.imshow(object_matrix_copy, cmap='Greys')
            plt.show()
        
        if show_both:
            c_mask = np.ma.masked_array(c_final, mask=object_matrix)

            plt.imshow(c_mask, cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.show()

        

# for smaller grids decrease omega
model = DLA(100,100,1.5)
print('initial point:',model.initial_point)
print('initial grid:\n',model.grid)
print('First diffusion: \n', model.SOR_Iteration())

# object = model.simulate_growth_model(steps=5)
# print('object coordinates are: ', object)
# print(model.SOR_Iteration())

model.visualize_DLA(steps=150, show_diffusion=True, show_object=True, show_both=True)
