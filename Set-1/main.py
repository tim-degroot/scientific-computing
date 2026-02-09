from math import sin, pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class vibrating_string:
    def __init__(self, f, L, c, dt, N, Nt, equation):
        self.f = f
        self.L = L
        self.c = c
        self.dt = dt
        self.N = N
        self.Nt = Nt
        self.equation = equation

        dx = L / N
        x = [i * dx for i in range(N + 1)]

        psi = np.zeros((N + 1, Nt + 1))

        for i in range(1, N):
            psi[i, 0] = f(x[i])

        for i in range(1, N):
            n = 1
            psi[i, n] = (
                c**2 * dt**2 / dx**2 * (psi[i + 1, n] - 2 * psi[i, n] + psi[i - 1, n])
                + 2 * psi[i, n]
                - psi[i, n - 1]
            )

        psi[0, 0] = 0
        psi[N, 0] = 0

        for n in range(1, Nt):
            psi[1:N, n + 1] = (
                c**2
                * dt**2
                / dx**2
                * (psi[2 : N + 1, n] - 2 * psi[1:N, n] + psi[0 : N - 1, n])
                + 2 * psi[1:N, n]
                - psi[1:N, n - 1]
            )
            psi[0, n + 1] = 0
            psi[N, n + 1] = 0

        self.x = x
        self.psi = psi

    def plot(self, filename=None, show=True):
        plt.figure()
        time_points = [
            0,
            self.Nt // 4,
            self.Nt // 2,
            3 * self.Nt // 4,
            self.Nt,
        ]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for t_idx, color in zip(time_points, colors):
            plt.plot(
                self.x, self.psi[:, t_idx], color=color, label=f"t={t_idx*self.dt:.3f}"
            )

        plt.xlabel(r"$x$")
        plt.ylabel(r"$\Psi(x_i, t_n)$")
        plt.title(f"Time evolution of vibrating string\n{self.equation}")
        plt.legend()

        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()

        plt.clf()

    def animate(self, interval=30, filename=None, show=True):
        fig, ax = plt.subplots()
        (line,) = ax.plot(self.x, self.psi[:, 0], color="tab:blue")
        ax.set_xlim(0, self.L)
        ax.set_ylim(np.min(self.psi), np.max(self.psi))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\Psi(x,t)$")
        ax.set_title(f"Vibrating string animation\n{self.equation}")

        def update(n):
            line.set_ydata(self.psi[:, n])
            ax.set_title(f"Vibrating string at t = {n*self.dt:.3f}\n{self.equation}")
            return (line,)

        anim = FuncAnimation(fig, update, frames=self.Nt + 1, interval=interval)

        if filename is not None:
            anim.save(filename)

        if show:
            plt.show()
        
        plt.clf()


if __name__ == "__main__":
    args = {"N": 100, "Nt": 200, "L": 1, "c": 1, "dt": 0.001}
    Bi = vibrating_string(
        f=lambda x: sin(2 * pi * x), equation=r"$\Psi(x,t=0)=\sin(2\pi x)$", **args
    )
    Bi.plot(filename="Bi.jpg", show=False)
    Bi.animate(filename="Bi.gif", show=False)

    Bii = vibrating_string(
        f=lambda x: sin(5 * pi * x), equation=r"$\Psi(x,t=0)=\sin(5\pi x)$", **args
    )
    Bii.plot(filename="Bii.jpg", show=False)
    Bii.animate(filename="Bii.gif", show=False)

    Biii = vibrating_string(
        f=lambda x: sin(5 * pi * x) if 2 / 5 > x > 1 / 5 else 0,
        equation=r"$\Psi(x,t=0)=\sin(5\pi x)$ if $1/5<x<2/5$, else $\Psi=0$",
        **args,
    )
    Biii.plot(filename="Biii.jpg", show=False)
    Biii.animate(filename="Biii.gif", show=False)
