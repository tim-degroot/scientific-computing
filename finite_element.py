from ngsolve import *
from netgen.occ import *

# viscosity
nu = 0.00115

# timestepping parameters
tau = 0.001
tend = 10

# Add specified Karman vortex
shape = Rectangle(2,0.41).Circle(0.2,0.2,0.05).Reverse().Face()
shape.edges.name="wall"
shape.edges.Min(X).name="inlet"
shape.edges.Max(X).name="outlet"

mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.07)).Curve(3)
Draw (mesh);


V = VectorH1(mesh,order=3, dirichlet="wall|cyl|inlet")
Q = H1(mesh,order=2)
X = V*Q

u,p = X.TrialFunction()
v,q = X.TestFunction()

stokes = nu*InnerProduct(grad(u), grad(v))+div(u)*q+div(v)*p - 1e-10*p*q
a = BilinearForm(X, symmetric=True)
a += stokes*dx
a.Assemble()

# nothing here ...
f = LinearForm(X)   
f.Assemble()

# gridfunction for the solution
gfu = GridFunction(X)

# parabolic inflow at inlet:
uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
inv_stokes = a.mat.Inverse(X.FreeDofs())

res = f.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += inv_stokes * res


# matrix for implicit Euler 
mstar = BilinearForm(X, symmetric=True)
mstar += (u*v + tau*stokes)*dx
mstar.Assemble()
inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

# the non-linear term 
conv = BilinearForm(X, nonassemble = True)
conv += (grad(u) * u) * v * dx

# for visualization
Draw (Norm(gfu.components[0]), mesh, "velocity", sd=3)

# implicit Euler/explicit Euler splitting method:
t = 0
with TaskManager():
    while t < tend:
        print ("t=", t, end="\r")

        conv.Apply (gfu.vec, res)
        res.data += a.mat*gfu.vec
        gfu.vec.data -= tau * inv * res    

        t = t + tau
        Redraw()