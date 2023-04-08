import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import diff
from neurodiffeq.ode import solve
from neurodiffeq.conditions import IVP

du = lambda u, t: diff(u, t) + u

init_val_ex = IVP(t_0=0.0, u_0=1.0)
solution_ex, loss_ex = solve(ode=du, condition=init_val_ex, t_min=0.0, t_max=2.0)
ts = np.linspace(0.0, 2.0, 50)
u_net = solution_ex(ts, to_numpy=True)
u_ana = np.exp(-ts)

plt.figure()
plt.plot(ts, u_net, label="ANN Based Solution")
plt.plot(ts, u_ana, ".", label="Analytical Solution")
plt.ylabel("u(t)")
plt.xlabel("t")
plt.title("ANN vs Analytical Solutions")
plt.legend()
plt.show()
plt.savefig("sol.png")