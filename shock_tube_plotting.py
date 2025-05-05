import numpy as np
import matplotlib.pyplot as plt

# Load numerical results from C code
data = np.loadtxt('shock_tube.txt')
x = data[:, 0]
rho = data[:, 1]
v = data[:, 2]
p = data[:, 3]

# Load exact solution (replace with your exact data file)
exact_data = np.loadtxt('exact_solution.txt')  # Format: x, rho_exact, v_exact, p_exact
x_exact = exact_data[:, 0]
rho_exact = exact_data[:, 1]
v_exact = exact_data[:, 2]
p_exact = exact_data[:, 3]

# Create figure with 3 subplots
plt.figure(figsize=(10, 8))

# Plot Density
plt.subplot(3, 1, 1)
plt.plot(x, rho, 'b-', linewidth=2, label='Numerical (Lax-Friedrichs)')
plt.plot(x_exact, rho_exact, 'r--', linewidth=1.5, label='Exact Solution')
plt.ylabel('Density ($\\rho$)')
plt.legend()
plt.grid(True)
plt.title('Shock Tube Problem at t=0.2 (Problem A)')

# Plot Velocity
plt.subplot(3, 1, 2)
plt.plot(x, v, 'b-', linewidth=2)
plt.plot(x_exact, v_exact, 'r--', linewidth=1.5)
plt.ylabel('Velocity ($v$)')
plt.grid(True)

# Plot Pressure
plt.subplot(3, 1, 3)
plt.plot(x, p, 'b-', linewidth=2)
plt.plot(x_exact, p_exact, 'r--', linewidth=1.5)
plt.ylabel('Pressure ($p$)')
plt.xlabel('Position ($x$)')
plt.grid(True)

plt.tight_layout()
plt.savefig('shock_tube_results.png', dpi=300)
plt.show()