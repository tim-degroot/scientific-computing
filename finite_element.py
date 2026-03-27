from ngsolve import *
from netgen.occ import *

class NavierStokesSolver:
    def __init__(self, nu=0.001, tau=0.001, tend=10, order=3):
        # Parameters
        self.nu = nu
        self.tau = tau
        self.tend = tend
        self.order = order
        
        # Initialize placeholders
        self.mesh = None
        self.X = None
        self.gfu = None
        self.inv_mstar = None
        self.a_mat = None
        self.conv = None

    def make_mesh(self, maxh=0.07, box_x=2, box_y=0.41, cyl_x=0.2, cyl_y=0.2, cyl_r=0.05):
        """Creates the geometry and mesh with named boundaries."""
        
        shape = Rectangle(box_x,box_y).Circle(cyl_x,cyl_y,cyl_r).Reverse().Face()
        shape.edges.name = "cyl"
        shape.edges.Min(X).name = "inlet"
        shape.edges.Max(X).name = "outlet"
        shape.edges.Min(Y).name = "wall"
        shape.edges.Max(Y).name = "wall"
                
        self.mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh)).Curve(3)

        return self.mesh

    def setup_spaces(self):
        """Defines Finite Element Spaces and GridFunctions."""
        V = VectorH1(self.mesh, order=self.order, dirichlet="wall|cyl|inlet")
        Q = H1(self.mesh, order=self.order-1) # Taylor-Hood stability
        self.X = V * Q
        self.gfu = GridFunction(self.X)
        
    def set_inflow(self, max_vel=1.5, height=0.41):
        """Defines the parabolic inflow profile."""
        uin = CoefficientFunction((max_vel * 4 * y * (height - y) / (height**2), 0))
        self.gfu.components[0].Set(uin, definedon=self.mesh.Boundaries("inlet"))

    def assemble_system(self):
        """Assembles the Stokes and Implicit Euler matrices."""
        u, p = self.X.TrialFunction()
        v, q = self.X.TestFunction()

        # Stokes Bilinear Form
        stokes = self.nu*InnerProduct(grad(u), grad(v)) + div(u)*q + div(v)*p - 1e-10*p*q
        
        # 1. Standard Stokes Matrix (for initial condition)
        a = BilinearForm(self.X, symmetric=True)
        a += stokes * dx
        a.Assemble()
        self.a_mat = a.mat

        # 2. M-Star Matrix for Time Stepping (Implicit Euler)
        mstar = BilinearForm(self.X, symmetric=True)
        mstar += (u*v + self.tau * stokes) * dx
        mstar.Assemble()
        self.inv_mstar = mstar.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")

        # 3. Non-linear Convection (Applied explicitly)
        self.conv = BilinearForm(self.X, nonassemble=True)
        self.conv += (grad(u) * u) * v * dx

    def solve_initial_stokes(self):
        """Solves the steady Stokes problem to provide a clean start."""
        f = LinearForm(self.X).Assemble()
        inv_stokes = self.a_mat.Inverse(self.X.FreeDofs())
        res = f.vec.CreateVector()
        res.data = f.vec - self.a_mat * self.gfu.vec
        self.gfu.vec.data += inv_stokes * res
        
    def get_drag_lift(self):
        """Calculates Drag and Lift coefficients on the 'cyl' boundary."""
        # Normal vector and Stress Tensor
        n = specialcf.normal(self.mesh.dim)
        u, p = self.gfu.components
        sigma = -p * Id(self.mesh.dim) + self.nu * (grad(u) + grad(u).trans)
                
        # Integrate forces over the cylinder surface
        # Assuming rho=1 and U_mean=1 for standard benchmark scaling
        fx = -Integrate((sigma * n)[0] * ds(definedon=self.mesh.Boundaries("cyl")), self.mesh)  # flip for normals
        fy = Integrate((sigma * n)[1] * ds(definedon=self.mesh.Boundaries("cyl")), self.mesh)
        
        # Reference values for DFG 2D-2 Benchmark: H=0.41, D=0.1, U_mean=1.0
        # Cd = 2 * fx / (rho * U_mean^2 * D)
        cd = 2 * fx / (1.0 * 1.0**2 * 0.1)
        cl = 2 * fy / (1.0 * 1.0**2 * 0.1)
        return cd, cl

    def run_simulation(self):
        """Executes the time-stepping loop."""
        
        times, drags, lifts = [], [], []
        t = 0
        res = self.gfu.vec.CreateVector()
        
        # for visualization
        Draw(Norm(self.gfu.components[0]), self.mesh, "velocity", sd=3)

        with TaskManager():
            while t < self.tend:
                self.conv.Apply(self.gfu.vec, res)
                res.data += self.a_mat * self.gfu.vec
                self.gfu.vec.data -= self.tau * self.inv_mstar * res
                
                # Record Data for Accuracy Analysis
                if int(t/self.tau) % 10 == 0:
                    cd, cl = self.get_drag_lift()
                    times.append(t)
                    drags.append(cd)
                    lifts.append(cl)
                
                t += self.tau
                Redraw()
                
        return times, drags, lifts 

# --- Example Usage ---
solver = NavierStokesSolver(nu=0.001, tau=0.0005, tend=5.0)
solver.make_mesh(maxh=0.07)
solver.setup_spaces()
solver.set_inflow(max_vel=1.5)
solver.assemble_system()
solver.solve_initial_stokes()
solver.run_simulation()