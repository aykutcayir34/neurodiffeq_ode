import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import diff
from neurodiffeq.ode import solve
from neurodiffeq.conditions import IVP
from neurodiffeq.monitors import Monitor1D

dduddt = lambda u, t: diff(u, t, order=2) + u
u = lambda t: np.sin(t)

init_val_ho = IVP(t_0=0.0, u_0=0, u_0_prime=1.0)
solution_ho, _ = solve(ode=dduddt, condition=init_val_ho, 
                       t_min=0.0, t_max=2*np.pi,
                       monitor=Monitor1D(t_min=0.0, t_max=2.0, check_every=100),
                       max_epochs=3000)
ts = np.linspace(0.0, 2*np.pi, 50)
u_net = solution_ho(ts, to_numpy=True)
u_ana = u(ts)

plt.figure()
plt.plot(ts, u_net, label="NN Based Solution")
plt.plot(ts, u_ana, ".", label="Analytical Solution")
plt.ylabel("u(t)")
plt.xlabel("t")
plt.title("Harmonic Oscillator")
plt.legend()
plt.savefig("harmonic_oscillator.png")
plt.show()
