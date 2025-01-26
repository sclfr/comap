import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
Lx, Ly = 1.0, 0.5  # Dimensions of the stair (1m x 0.5m)
Nx, Ny = 100, 50  # Number of grid points in x and y directions
T = 200  # Total time steps
dt = 0.4  # Time step
alpha = 0.01  # Background wear rate
U = 5.0  # Usage rate (constant)
q = 0.5  # Wear per step coefficient

# Create a 2D grid for positions
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Define the stepping probability distribution P(x, y)
sigma_x, sigma_y = 0.12, 0.12
P_up_y = np.exp(-((Y - 0.00 * Ly)**2) / (2 * (sigma_y)**2))
P_down_y = np.exp(-((Y - 0.6 * Ly)**2) / (2 * sigma_y**2))
P_single_x = np.exp(-((X - 0.35 * Lx)**2) / (2 * sigma_x**2)) + \
             np.exp(-((X - 0.65 * Lx)**2) / (2 * sigma_x**2))
P_pairs_x = np.exp(-((X - 0.3 * Lx)**2) / (2 * sigma_x**2)) + \
            np.exp(-((X - 0.45 * Lx)**2) / (2 * sigma_x**2)) + \
            np.exp(-((X - 0.55 * Lx)**2) / (2 * sigma_x**2)) + \
            np.exp(-((X - 0.7 * Lx)**2) / (2 * sigma_x**2))
P0 = (P_up_y + P_down_y) * (P_single_x + P_pairs_x)
P0 /= np.sum(P0)  # Normalize to make it a valid probability distribution


# Prepare the figure for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Adjustments to parameters and initial conditions
Q = np.full((Ny, Nx), 1e-4)  # Initialize wear with small values to avoid stagnation
h = np.ones((Ny, Nx))  # Initial height of the stair remains unchanged

# Update function with adjusted dynamics
def update_with_qbar_adjusted(t):
    global Q, h
    ax.clear()
    Q_bar = np.mean(Q) + 1e-6  # Regularized mean wear to avoid division by zero
    dQ = dt * (alpha + U * q * P0) * Q / Q_bar  # Compute change in wear with Q/Qbar
    Q += dQ  # Update wear
    h -= dQ  # Update height

    # Plot the updated height
    ax.plot_surface(X, Y, h, cmap='viridis', edgecolor='none')
    ax.set_title(f"Wear Evolution at Time {t*dt:.2f}")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (h)')
    ax.set_box_aspect([8,4,4])
    ax.set_zlim(0, 1)  # Keep z-axis limits fixed for consistent visualization

# Create the updated animation with adjusted parameters
anim_with_qbar_adjusted = FuncAnimation(fig, update_with_qbar_adjusted, frames=int(T/dt), interval=50)

# Display the animation
plt.show()

