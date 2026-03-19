# Assignment Set 3

Most important: always motivate and argue your choices well, for this assignment the train of thought, the techniques you apply, and the reason why you are applying them is much more important than the numerical value of the results. Results without proper motivation will not be accepted.

At the end of the course I will publish the ranking among the teams:

- Highest achieved numerically stable Re.
- Highest WiFi signal strength.

Good luck!

## 3.1 Solving the Navier-Stokes equation

At the heart of fluid dynamics lies the incompressible Navier-Stokes equations, a set of PDEs that describe the motion of fluid-like substances. You have seen this at the second guided self-study. For a fluid with constant density $\rho$ and viscosity $\nu$, the equations are expressed as:

$$\begin{align}
\frac{\partial\mathbf{u}}{\partial t}+ (\mathbf{u}\cdot\nabla)\mathbf{u}=-\frac{1}{\rho}\nabla p+\nu\nabla^2\mathbf{u}, \\
\nabla\cdot\mathbf{u}=0,
\end{align}$$

where $\mathbf{u}$ is the velocty vector and $p$ is the pressure.

In both industry and research, these equations are used to model everything from the aerodynamic lift of an aircraft wing and the cooling of microelectronics to complex blood flow through human arteries. Because analytical (exact) solutions to these equations exist only for the simplest geometries, we rely entirely on numerical solutions. The accuracy and stability of these solutions depend heavily on the chosen discretization method, the resolution of the mesh, and the handling of the non-linear convective term $(\mathbf{u}\cdot\nabla)\mathbf{u}$.

It is very important to understand that several different solution options might exist for the same mathematical model. Understanding the benefits, weaknesses, and limits of these options allows you to identify the best solution in a given problem context. In this challenge, you will implement the same flow problem using three different numerical methods: 1. finite difference, 2. finite element (using ngsolve), 3. lattice Boltzmann.

To validate our numerical solvers, we frequently look to the Kármán vortex street. This phenomenon occurs when a fluid flows past a bluff body (such as a cylinder) at specific Reynolds numbers, resulting in a repeating pattern of swirling vortices caused by the unsteady separation of flow. It serves as a classic benchmark in computational fluid dynamics (CFD) because it tests a solver’s ability to capture time-dependent oscillations and wake dynamics accurately.

Beyond the lab, Kármán vortex streets occur naturally on a massive scale; they are often visible in satellite imagery as clouds drift past high-altitude islands, and they are the physical cause behind the "singing" of power lines in the wind or the structural vibrations of underwater cables.

**Challenge A.** *(5 points)* Implement the Kármán vortex street following the setup below, test for correctness, then make it break! Test the limits of each of the three methods: how high can you go with Re and stay stable?

While trying to reach the highest Re you can, build and present your own opinion on the three methods! Here are some ideas on what you can compare to get you started (and you are invited to bring your own ideas too): How does the computational cost compare? How does the accuracy and/or numerical stability scale with the mesh resolution? How does accuracy differ very close to curved objects? How easy is it to produce an implementation? Is it suitable for parallel execution? Would it fit GPU execution?

## 3.2 Optimizing the position of the WiFi router

Here you will apply the Helmholtz equation to a practical, modern challenge: ap-
proximating WiFi signal strength within a complex indoor environment. WiFi sig-
nals are electromagnetic waves that oscillate at high frequencies (typically 2.4 GHz
or 5 GHz). While full Maxwell equations are required for high-fidelity physics, the
scalar Helmholtz equation provides an excellent approximation for mapping signal
coverage in a 2D floor plan:

$$\begin{equation}
\Delta u+k^2u=f,
\end{equation}$$

where $u$ is the complex wave field, $k=\omega/c$ is the wavenumber (scaled in this document for visualization), and $f$ is a Gaussian pulse source at the router position.

The source term $f$ in this challenge can be modeled as a Gaussian pulse centered at the router position $(x_r,y_r)$:

$$\begin{equation}
f(x,y)=A\cdot\exp\left(-\frac{(x-x_r)^2+(y-y_r)^2}{2\sigma^2}\right)
\end{equation}$$

where the amplitude $A=10^4$ controls the overall signal strength, and the width $\sigma = 0.2$ m determines the spatial extent of the source (40 cm diameter at half-maximum).

Side-note: This represents a localized, omnidirectional point source. The Gaussian shape provides smooth derivatives for numerical stability while approximating a point emitter.

**Challenge B.** *(5 points)* Find the best position for the router that yields the highest signal strength summed over the given measurement points!